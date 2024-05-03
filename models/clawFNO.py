import torch.nn.functional as F
import torch
import torch.nn as nn
from utils import grid, fc


# ----------------------------------------------------------------------------------------------------------------------
# Baseline FNO: code from https://github.com/neural-operator/fourier_neural_operator
# ----------------------------------------------------------------------------------------------------------------------

################################################################
# Normalizer: code from https://github.com/zongyi-li/fourier_neural_operator
################################################################
# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


################################################################
# 2D fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class clawFNO2d(nn.Module):
    def __init__(self, num_channels, modes1, modes2, width, initial_step, grid_type):
        super(clawFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic
        self.fc_d, self.fc_C, self.fc_filter = 3, 25, False
        self.fc_pad = fc(fc_d=self.fc_d, fc_C=self.fc_C)

        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.grid = grid(twoD=True, grid_type=grid_type)
        self.p = nn.Linear(initial_step * num_channels + self.grid.grid_dim, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP2d(self.width, self.width, self.width)
        self.mlp1 = MLP2d(self.width, self.width, self.width)
        self.mlp2 = MLP2d(self.width, self.width, self.width)
        self.mlp3 = MLP2d(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP2d(self.width, num_channels - 1, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        grids = self.get_grid(x.shape).to(x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.grid(x)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        # x = x.permute(0, 2, 3, 1)

        n_pts_x, n_pts_y = x.shape[-2], x.shape[-1]
        domain_size_x = torch.max(grids[..., 0]) - torch.min(grids[..., 0])
        domain_size_y = torch.max(grids[..., 1]) - torch.min(grids[..., 1])
        # compute derivatives via fft padded by fc
        x, fc_prd_y = self.fc_pad(x, domain_size=domain_size_y)
        x = x.permute(0, 1, 3, 2)
        x, fc_prd_x = self.fc_pad(x, domain_size=domain_size_x)
        x = x.permute(0, 1, 3, 2)

        # compute wave number (i.e., spatial freq.)
        fc_npoints_total_x = n_pts_x + self.fc_C
        fc_npoints_total_y = n_pts_y + self.fc_C
        kx = torch.fft.fftfreq(fc_npoints_total_x).to(x.device) * 2 * torch.pi * fc_npoints_total_x / fc_prd_x
        ky = torch.fft.fftfreq(fc_npoints_total_y).to(x.device) * 2 * torch.pi * fc_npoints_total_y / fc_prd_y
        # get reciprocal grids (wave number axes)
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')

        # compute derivatives
        x_ft = torch.fft.fftn(x, dim=[-2, -1])
        if self.fc_filter:
            # filter spectral coefficients to eliminate growth of high-freq errors
            fc_alpha, fc_p = 10, 14
            fc_filter_coeffs_x = torch.exp(
                -fc_alpha * (2. * torch.fft.fftfreq(fc_npoints_total_x).to(x.device)) ** fc_p)
            fc_filter_coeffs_y = torch.exp(
                -fc_alpha * (2. * torch.fft.fftfreq(fc_npoints_total_y).to(x.device)) ** fc_p)
            dAdx = torch.real(
                torch.fft.ifftn(1j * Kx * fc_filter_coeffs_x * x_ft, s=(x.size(-2), x.size(-1)))
                )[..., :-self.fc_C, :-self.fc_C].permute(0, 2, 3, 1)
            dAdy = torch.real(
                torch.fft.ifftn(1j * Ky * fc_filter_coeffs_y * x_ft, s=(x.size(-2), x.size(-1)))
                )[..., :-self.fc_C, :-self.fc_C].permute(0, 2, 3, 1)
        else:
            dAdx = torch.real(torch.fft.ifftn(1j * Kx * x_ft, s=(x.size(-2), x.size(-1)))
                              )[..., :-self.fc_C, :-self.fc_C].permute(0, 2, 3, 1)
            dAdy = torch.real(torch.fft.ifftn(1j * Ky * x_ft, s=(x.size(-2), x.size(-1)))
                              )[..., :-self.fc_C, :-self.fc_C].permute(0, 2, 3, 1)

        # assume 12
        x = torch.cat((dAdy, -dAdx), dim=-1)

        return x.unsqueeze(-2)

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)


################################################################
# 3d fourier layers
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP3d, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class clawFNO3d(nn.Module):
    def __init__(self, num_channels, modes1, modes2, modes3, width, initial_step, grid_type, time, time_pad=False):
        super(clawFNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.time = time
        self.time_pad = time_pad
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc_d, self.fc_C, self.fc_filter = 3, 25, False
        self.fc_pad = fc(fc_d=self.fc_d, fc_C=self.fc_C)
        self.grid = grid(twoD=False, grid_type=grid_type)

        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.p = nn.Linear(initial_step * num_channels + self.grid.grid_dim, self.width)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP3d(self.width, self.width, self.width)
        self.mlp1 = MLP3d(self.width, self.width, self.width)
        self.mlp2 = MLP3d(self.width, self.width, self.width)
        self.mlp3 = MLP3d(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP3d(self.width, num_channels, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        # change data from (u,v,h) to (hu,hv,h)
        # x[..., 0] *= x[..., 2]
        # x[..., 1] *= x[..., 2]

        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        x = self.grid(x)
        grids = self.get_grid(x.shape).to(x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)

        if self.time and self.time_pad:
            x = F.pad(x, (0, self.padding))  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        if self.time and self.time_pad:
            x = x[..., :-self.padding]
        x = self.q(x)
        # x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic

        n_pts_x, n_pts_y, n_pts_z = x.shape[-3], x.shape[-2], x.shape[-1]
        domain_size_x = torch.max(grids[..., 0]) - torch.min(grids[..., 0])
        domain_size_y = torch.max(grids[..., 1]) - torch.min(grids[..., 1])
        domain_size_z = torch.max(grids[..., 2]) - torch.min(grids[..., 2])
        # compute derivatives via fft padded by fc
        x, fc_prd_z = self.fc_pad(x, domain_size=domain_size_z)
        x = x.permute(0, 1, 3, 4, 2)
        x, fc_prd_x = self.fc_pad(x, domain_size=domain_size_x)
        x = x.permute(0, 1, 4, 3, 2)
        x, fc_prd_y = self.fc_pad(x, domain_size=domain_size_y)
        x = x.permute(0, 1, 2, 4, 3)

        # compute wave number (i.e., spatial freq.)
        fc_npoints_total_x = n_pts_x + self.fc_C
        fc_npoints_total_y = n_pts_y + self.fc_C
        fc_npoints_total_z = n_pts_z + self.fc_C
        kx = torch.fft.fftfreq(fc_npoints_total_x).to(x.device) * 2 * torch.pi * fc_npoints_total_x / fc_prd_x
        ky = torch.fft.fftfreq(fc_npoints_total_y).to(x.device) * 2 * torch.pi * fc_npoints_total_y / fc_prd_y
        kz = torch.fft.fftfreq(fc_npoints_total_z).to(x.device) * 2 * torch.pi * fc_npoints_total_z / fc_prd_z
        # get reciprocal grids (wave number axes)
        Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')

        # compute derivatives
        x_ft = torch.fft.fftn(x, dim=[-3, -2, -1])
        if self.fc_filter:
            # filter spectral coefficients to eliminate growth of high-freq errors
            fc_alpha, fc_p = 10, 14
            fc_filter_coeffs_x = torch.exp(
                -fc_alpha * (2. * torch.fft.fftfreq(fc_npoints_total_x).to(x.device)) ** fc_p)
            fc_filter_coeffs_y = torch.exp(
                -fc_alpha * (2. * torch.fft.fftfreq(fc_npoints_total_y).to(x.device)) ** fc_p)
            fc_filter_coeffs_z = torch.exp(
                -fc_alpha * (2. * torch.fft.fftfreq(fc_npoints_total_z).to(x.device)) ** fc_p)
            dAdx = torch.real(torch.fft.ifftn(1j * Kx * fc_filter_coeffs_x * x_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
                              )[..., :-self.fc_C, :-self.fc_C, :-self.fc_C].permute(0, 2, 3, 4, 1)
            dAdy = torch.real(torch.fft.ifftn(1j * Ky * fc_filter_coeffs_y * x_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
                              )[..., :-self.fc_C, :-self.fc_C, :-self.fc_C].permute(0, 2, 3, 4, 1)
            dAdz = torch.real(torch.fft.ifftn(1j * Kz * fc_filter_coeffs_z * x_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
                              )[..., :-self.fc_C, :-self.fc_C, :-self.fc_C].permute(0, 2, 3, 4, 1)
        else:
            dAdx = torch.real(torch.fft.ifftn(1j * Kx * x_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
                              )[..., :-self.fc_C, :-self.fc_C, :-self.fc_C].permute(0, 2, 3, 4, 1)
            dAdy = torch.real(torch.fft.ifftn(1j * Ky * x_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
                              )[..., :-self.fc_C, :-self.fc_C, :-self.fc_C].permute(0, 2, 3, 4, 1)
            dAdz = torch.real(torch.fft.ifftn(1j * Kz * x_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
                              )[..., :-self.fc_C, :-self.fc_C, :-self.fc_C].permute(0, 2, 3, 4, 1)

        # assume 12, 13, 23
        x = torch.cat((dAdy[..., 0:1] + dAdz[..., 1:2],
                       -dAdx[..., 0:1] + dAdz[..., 2:3],
                       -dAdx[..., 1:2] - dAdy[..., 2:3]), dim=-1)

        # check if learned x is div free

        # change data back to (u,v,h) from (hu,hv,h)
        # x[..., 0] /= x[..., 2].clone()
        # x[..., 1] /= x[..., 2].clone()
        # x[..., 0] = x[..., 0].clone() / x[..., 2].clone()
        # x[..., 1] = x[..., 1].clone() / x[..., 2].clone()

        # if (x[..., 2] == 0).any().item():
        #     print('>> Warning: at least one denominator is zero!')

        if not self.time:  # add a time axis
            x = x.unsqueeze(-2)
        return x

    def get_grid(self, shape):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        # gridx = torch.linspace(0, 1, size_x)
        gridx = torch.linspace(-2.5, 2.5, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        # gridy = torch.linspace(0, 1, size_y)
        gridy = torch.linspace(-2.5, 2.5, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)

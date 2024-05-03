import torch
import numpy as np
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from scipy.stats import pearsonr


################################################################
# Dataset class
################################################################
class pde_data(torch.utils.data.Dataset):
    def __init__(self, data, T_in, T_out=None, train=True, strategy="markov", std=0.0):
        self.markov = strategy == "markov"
        self.teacher_forcing = strategy == "teacher_forcing"
        self.one_shot = strategy == "oneshot"
        self.data = data[..., :(T_in + T_out)] if self.one_shot else data[..., :(T_in + T_out), :]
        self.nt = T_in + T_out
        self.T_in = T_in
        self.T_out = T_out
        self.num_hist = 1 if self.markov else self.T_in
        self.train = train
        self.noise_std = std

    def __len__(self):
        if self.train:
            if self.markov:
                return len(self.data) * (self.nt - 1)
            if self.teacher_forcing:
                return len(self.data) * (self.nt - self.T_in)
        return len(self.data)

    def __getitem__(self, idx):
        if not self.train or not (self.markov or self.teacher_forcing):  # full target: return all future steps
            pde = self.data[idx]
            if self.one_shot:
                x = pde[..., :self.T_in, :]
                x = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                y = pde[..., self.T_in:(self.T_in + self.T_out), :]
            else:
                x = pde[..., (self.T_in - self.num_hist):self.T_in, :]
                y = pde[..., self.T_in:(self.T_in + self.T_out), :]
            return x, y
        pde_idx = idx // (self.nt - self.num_hist)  # Markov / teacher forcing: only return one future step
        t_idx = idx % (self.nt - self.num_hist) + self.num_hist
        pde = self.data[pde_idx]
        x = pde[..., (t_idx - self.num_hist):t_idx, :]
        y = pde[..., t_idx, :]
        if self.noise_std > 0:
            x += torch.randn(*x.shape, device=x.device) * self.noise_std

        return x, y


################################################################
# Lploss: code from https://github.com/zongyi-li/fourier_neural_operator
################################################################
# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        assert x.shape == y.shape and len(x.shape) == 3, ">> Error: wrong tensor shape to compute loss!"
        diff_norms = torch.norm(x - y, self.p, 1)
        y_norms = torch.norm(y, self.p, 1)

        if self.reduction:
            loss = (diff_norms / y_norms).mean(-1)  # average over channel dimension
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


################################################################
# equivariance checks
################################################################
# function for checking equivariance to 90 rotations of a scalar field
def eq_check_rt(model, x, spatial_dims):
    model.eval()
    diffs = []
    with torch.no_grad():
        out = model(x)
        out[out == 0] = float("nan")
        for j in range(len(spatial_dims)):
            for l in range(j + 1, len(spatial_dims)):
                dims = [spatial_dims[j], spatial_dims[l]]
                diffs.append([((out.rot90(k=k, dims=dims) - model(x.rot90(k=k, dims=dims))) / out.rot90(k=k,
                                                                                                        dims=dims)).abs().nanmean().item() * 100
                              for k in range(1, 4)])
    return torch.tensor(diffs).mean().item()


# function for checking equivariance to reflections of a scalar field
def eq_check_rf(model, x, spatial_dims):
    model.eval()
    diffs = []
    with torch.no_grad():
        out = model(x)
        out[out == 0] = float("nan")
        for j in spatial_dims:
            diffs.append(
                ((out.flip(dims=(j,)) - model(x.flip(dims=(j,)))) / out.flip(dims=(j,))).abs().nanmean().item() * 100)
    return torch.tensor(diffs).mean().item()


################################################################
# grids
################################################################
class grid(torch.nn.Module):
    def __init__(self, twoD, grid_type):
        super(grid, self).__init__()
        assert grid_type in ["cartesian", "symmetric", "None"], "Invalid grid type"
        self.symmetric = grid_type == "symmetric"
        self.include_grid = grid_type != "None"
        self.grid_dim = (1 + (not self.symmetric) + (not twoD)) * self.include_grid
        if self.include_grid:
            if twoD:
                self.get_grid = self.twoD_grid
            else:
                self.get_grid = self.threeD_grid
        else:
            self.get_grid = torch.nn.Identity()

    def forward(self, x):
        return self.get_grid(x)

    def twoD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy), dim=-1)
        else:
            midx = midy = 0.5
            # midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = gridx + gridy
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)

    def threeD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        # gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        # gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridx = torch.linspace(-2.5, 2.5, size_x).reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(-2.5, 2.5, size_y).reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z).reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy, gridz), dim=-1)
        else:
            # midx = 0.5
            # midy = (size_y - 1) / (2 * (size_x - 1))
            midx = midy = 0.
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = torch.cat((gridx + gridy, gridz), dim=-1)
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)


################################################################
# fourier continuation for fft and ders on non-periodic data
################################################################
class fc(object):
    def __init__(self, fc_d=3, fc_C=25):
        super(fc, self).__init__()

        self.fc_d = fc_d
        self.fc_C = fc_C

        fc_data_pairs = ((3, 6), (3, 12), (3, 25), (5, 25))
        assert (fc_d, fc_C) in fc_data_pairs, ">> Error: precomputed FC matrices not found for the input degree!"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.fc_d == 3 and self.fc_C == 25:
            fc_A = torch.tensor([[0.577350269189425, 1.41421356237007, 4.08248290457551],
                                 [0.577350268932529, 2.12132033966420, 10.2062071802068],
                                 [0.577350221019144, 2.82842639268168, 18.7794060152318],
                                 [0.577347652221430, 3.53549399024348, 29.8012851068571],
                                 [0.577291059875212, 4.24173355523443, 43.2551322460336],
                                 [0.576658953051786, 4.93909676130202, 58.9693320070330],
                                 [0.572570029004639, 5.58268162484116, 75.9760350811826],
                                 [0.555837186090934, 6.02703866329299, 91.0846940619291],
                                 [0.509831647972551, 6.00048596828142, 98.1277933706341],
                                 [0.421270599722172, 5.26159744521892, 91.0174238476298],
                                 [0.298559139312803, 3.88635368624552, 69.8896198069309],
                                 [0.174035832724397, 2.33081834487773, 43.0524050354295],
                                 [0.0806753357082983, 1.10206203421019, 20.7407332422353],
                                 [0.0288930418776341, 0.400289045332448, 7.63521737660355],
                                 [0.00777163554111367, 0.108783666634782, 2.09560532833085],
                                 [0.00152109005402703, 0.0214568816355638, 0.416455939791407],
                                 [0.000208380234960350, 0.00295706340634324, 0.0577289864413102],
                                 [1.89994028313830e-05, 0.000270889262028986, 0.00531296816181229],
                                 [1.07736236480021e-06, 1.54193275128022e-05, 0.000303559104610551],
                                 [3.45792118873408e-08, 4.96450646712166e-07, 9.80397286157291e-06],
                                 [5.48560209319221e-10, 7.89621720928448e-09, 1.56342203350098e-07],
                                 [3.50735040678574e-12, 5.05981661205677e-11, 1.00404742641513e-09],
                                 [6.52264500069445e-15, 9.42766487410956e-14, 1.87435770403075e-12],
                                 [1.99918217690639e-18, 2.89433588113721e-17, 5.76393682469331e-16],
                                 [3.27190167760519e-23, 4.74378753707539e-22, 9.46086295849135e-21]])
            fc_Q = torch.tensor([[0.577350269189626, -0.707106781186548, 0.408248290463863],
                                 [0.577350269189626, 0, -0.816496580927726],
                                 [0.577350269189626, 0.707106781186548, 0.408248290463863]])
        elif self.fc_d == 3 and self.fc_C == 6:
            fc_A = torch.tensor([[0.483868985307726, 0.964483282153122, 1.45399048686443],
                                 [0.327637155684739, 0.837213869787502, 1.88530013923995],
                                 [0.157342696770378, 0.457454312621708, 1.23309224217555],
                                 [0.0473720802649667, 0.148437703116213, 0.442738391766524],
                                 [0.00712731551515447, 0.0234174352696140, 0.0744836680937376],
                                 [0.000305680893875259, 0.00103738994246047, 0.00344990240574944]])
            fc_Q = torch.tensor([[0.577350269189626, -0.707106781186571, 0.408248290469261],
                                 [0.577350269189626, -2.35513868802566e-14, -0.816496580922371],
                                 [0.577350269189626, 0.707106781186524, 0.408248290469176]])
        elif self.fc_d == 3 and self.fc_C == 12:
            fc_A = torch.tensor([[0.573546998158633, 1.38271879584259, 3.56054049955439],
                                 [0.556560125288930, 1.94492807228321, 7.47317548872105],
                                 [0.510066046718126, 2.24265998438925, 10.1475854187773],
                                 [0.424298914594900, 2.16089107636261, 10.1986450056090],
                                 [0.308443695632726, 1.72956553670865, 7.85073632070237],
                                 [0.189194505600595, 1.13212135790947, 4.61894020810674],
                                 [0.0943851736050239, 0.591158561877336, 2.01548026040243],
                                 [0.0365633586637526, 0.236821168912075, 0.607959970913247],
                                 [0.0102748722786736, 0.0683057678956682, 0.107245988674933],
                                 [0.00187121305539315, 0.0127093245213994, 0.00528234785760654],
                                 [0.000178691074999025, 0.00123687092094845, -0.00113592686301733],
                                 [5.54545715050955e-06, 3.90864060407427e-05, -9.70937307549835e-05]])
            fc_Q = torch.tensor([[0.577350269189626, -0.707106781186571, 0.408248290469261],
                                 [0.577350269189626, -2.35513868802566e-14, -0.816496580922371],
                                 [0.577350269189626, 0.707106781186524, 0.408248290469176]])
        elif self.fc_d == 5 and self.fc_C == 25:
            fc_A = torch.tensor(
                [[0.447213595213893, 0.948683295425315, 1.87082866551978, 4.42718838564417, 15.0598750754598],
                 [0.447213566471397, 1.26491079655245, 3.74165453387796, 13.2815313368687, 65.2589232783266],
                 [0.447212450573684, 1.58112823059934, 6.14689495510080, 28.4591041307591, 179.978922289879],
                 [0.447191374730374, 1.89715982691718, 9.08465342229828, 51.5175937512142, 396.488186950784],
                 [0.446966918238617, 2.21128543843541, 12.5362266932715, 83.8049115386688, 756.975150786320],
                 [0.445479186233559, 2.51347442016254, 16.3914387996148, 125.512986200577, 1292.74214074293],
                 [0.438908561109676, 2.76710296179998, 20.2420970314981, 172.993074162615, 1978.10474822220],
                 [0.418619122985368, 2.88754975451463, 23.1220571972341, 215.119536388226, 2663.08991751941],
                 [0.373300556399696, 2.75856449291266, 23.6344830594980, 234.093637832804, 3070.54159967224],
                 [0.298335785994355, 2.31825947632122, 20.8522387775748, 215.990933668143, 2952.25128032015],
                 [0.205072820092728, 1.65156463965807, 15.3760499988086, 164.403004449191, 2313.95761273702],
                 [0.117049714956061, 0.966895368460470, 9.22502263196689, 100.901672765237, 1450.52467458190],
                 [0.0538541955574408, 0.453065603931456, 4.39999966589100, 48.9319254026378, 714.482617678240],
                 [0.0194453279835376, 0.165805429884090, 1.63157769459152, 18.3717245751179, 271.440908271919],
                 [0.00536208084274965, 0.0461885649130693, 0.459094234574135, 5.21916746800331, 77.8230392186869],
                 [0.00109492152815563, 0.00950637133551385, 0.0952341334581193, 1.09087064775007, 16.3851178230366],
                 [0.000159433272131088, 0.00139297290432927, 0.0140428463158209, 0.161840379168189, 2.44536200073336],
                 [1.57701275490821e-05, 0.000138492042265180, 0.00140339177869927, 0.0162554258102865,
                  0.246829722357186],
                 [9.93148306914283e-07, 8.75900860591491e-06, 8.91421685490032e-05, 0.00103691256056054,
                  0.0158107748841144],
                 [3.64187707221359e-08, 3.22354787675283e-07, 3.29273616098822e-06, 3.84404683743866e-05,
                  0.000588242397881924],
                 [6.84445515750482e-10, 6.07711557009970e-09, 6.22727773843944e-08, 7.29280794087538e-07,
                  1.11948617427213e-05],
                 [5.44798260931865e-12, 4.85034977964738e-11, 4.98403470395195e-10, 5.85298754658622e-09,
                  9.00944917393720e-08],
                 [1.35744246961945e-14, 1.21143860339078e-13, 1.24790124441774e-12, 1.46907623627354e-11,
                  2.26690195327967e-10],
                 [6.28970107833331e-18, 5.62526369280258e-17, 5.80740395643517e-16, 6.85182118735231e-15,
                  1.05963638253108e-13],
                 [1.96148342759244e-22, 1.75768510785428e-21, 1.81823606968664e-20, 2.14954096308993e-19,
                  3.33098142686937e-18]])
            fc_Q = torch.tensor(
                [[0.447213595499958, -0.632455532033676, 0.534522483824849, -0.316227766016838, 0.119522860933439],
                 [0.447213595499958, -0.316227766016838, -0.267261241912424, 0.632455532033676, -0.478091443733757],
                 [0.447213595499958, 0, -0.534522483824849, 0, 0.717137165600636],
                 [0.447213595499958, 0.316227766016838, -0.267261241912424, -0.632455532033676, -0.478091443733757],
                 [0.447213595499958, 0.632455532033676, 0.534522483824849, 0.316227766016838, 0.119522860933439]])

        fc_F1 = torch.flipud(torch.eye(self.fc_d))
        fc_F2 = torch.flipud(torch.eye(self.fc_C))

        self.fc_AQ = torch.matmul(fc_A, fc_Q.T).to(device)
        self.fc_AlQl = torch.matmul(torch.matmul(torch.matmul(fc_F2, fc_A), fc_Q.T), fc_F1).to(device)

    def fc_pad(self, x, domain_size=1.0):
        n_pts_x, n_pts_y, n_pts_z = x.shape[-3], x.shape[-2], x.shape[-1]
        # self.fc_A, self.fc_Q = self.fc_A.to(x.device), self.fc_Q.to(x.device)

        fc_h = domain_size / (n_pts_z - 1)
        fc_npoints_total = n_pts_z + self.fc_C
        fc_prd = fc_npoints_total * fc_h

        x = torch.cat((x, torch.matmul(self.fc_AlQl, x[..., :self.fc_d].unsqueeze(-1)).squeeze(-1)
                       + torch.matmul(self.fc_AQ, x[..., -self.fc_d:].unsqueeze(-1)).squeeze(-1)), dim=-1)

        return x, fc_prd

    def __call__(self, x, domain_size=1.0):
        return self.fc_pad(x, domain_size)


# function to check if a 3D field is divergence free
def div_free_2d_check(u, v):
    u_orig = u.clone()
    v_orig = v.clone()

    batchsize, size_x, size_y, size_z = u.shape[0], u.shape[1], u.shape[2], u.shape[3]
    u = u.permute(0, 3, 2, 1)
    v = v.permute(0, 3, 1, 2)

    experiment = 'ns'
    # get grids
    if experiment == 'ns':
        gridx = torch.linspace(0, 1, size_x)
        gridy = torch.linspace(0, 1, size_y)
        domain_size_x = domain_size_y = 1
    else:
        gridx = torch.linspace(0, 1, size_x)
        gridy = torch.linspace(0, 1, size_y)
        domain_size_x = domain_size_y = 1

    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    grid = torch.cat((gridx, gridy), dim=-1)

    n_pts_x, n_pts_y = u.shape[-1], u.shape[-2]
    fc_d, fc_C, fc_filter = 3, 25, False
    fc_pad = fc(fc_d=fc_d, fc_C=fc_C)
    u, fc_prd_u = fc_pad(u, domain_size=domain_size_x)
    u = u.permute(0, 1, 3, 2)
    v, fc_prd_v = fc_pad(v, domain_size=domain_size_y)

    # u
    fc_npoints_total = n_pts_x + fc_C
    kx = torch.fft.fftfreq(fc_npoints_total).to(u.device) * 2 * torch.pi * fc_npoints_total / fc_prd_u
    ky = torch.fft.fftfreq(n_pts_y).to(u.device) * 2 * torch.pi * n_pts_y / (grid[0, 0, -1, 1] - grid[0, 0, 0, 1])
    # get reciprocal grids (wave number axes)
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
    u_ft = torch.fft.fftn(u, dim=[-2, -1])
    dudx = torch.real(torch.fft.ifftn(1j * Kx * u_ft, s=(u.size(-2), u.size(-1)))
                      )[:, :, :-fc_C, :].permute(0, 2, 3, 1)

    # v
    fc_npoints_total = n_pts_y + fc_C
    kx = torch.fft.fftfreq(n_pts_x).to(u.device) * 2 * torch.pi * n_pts_x / (grid[0, -1, 0, 0] - grid[0, 0, 0, 0])
    ky = torch.fft.fftfreq(fc_npoints_total).to(u.device) * 2 * torch.pi * fc_npoints_total / fc_prd_v
    # get reciprocal grids (wave number axes)
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
    v_ft = torch.fft.fftn(v, dim=[-2, -1])
    dvdy = torch.real(torch.fft.ifftn(1j * Ky * v_ft, s=(v.size(-2), v.size(-1)))
                      )[:, :, :, :-fc_C].permute(0, 2, 3, 1)

    i_div_free = dudx + dvdy
    i_plot_data = 0
    if i_plot_data:
        i_time = -2
        i_sample = 0
        plt.ioff()
        plt.rcParams["font.family"] = "Times New Roman"
        interp_coeff, interp_res = 'none', 'spline16'
        fig, ax = plt.subplots(1, 3, figsize=(6, 6), subplot_kw={'xticks': [], 'yticks': []})
        # gt_u = ax[0].imshow(dudx[0, :, :, 5], origin='lower', aspect='auto', interpolation=interp_res)
        # gt_u = ax[0].imshow(dudx[0, :, :, 5], interpolation=interp_coeff)
        gt_u = ax[0].imshow(dudx[i_sample, :, :, i_time], interpolation=interp_coeff)
        fig.colorbar(gt_u, ax=ax[0], shrink=0.4)
        # gt_v = ax[1].imshow(dvdy[0, :, :, 5], origin='lower', aspect='auto', interpolation=interp_res)
        # gt_v = ax[1].imshow(dvdy[0, :, :, 5], interpolation=interp_coeff)
        gt_v = ax[1].imshow(-dvdy[i_sample, :, :, i_time], interpolation=interp_coeff)
        fig.colorbar(gt_v, ax=ax[1], shrink=0.4)
        # gt_div = ax[2].imshow(i_div_free[0, :, :, 5], origin='lower', aspect='auto', interpolation=interp_res)
        # gt_div = ax[2].imshow(i_div_free[0, :, :, 5], interpolation=interp_coeff)
        gt_div = ax[2].imshow(i_div_free[i_sample, :, :, i_time], interpolation=interp_coeff)
        fig.colorbar(gt_div, ax=ax[2], shrink=0.4)
        cbar_min, cbar_max = torch.min(dudx[i_sample, :, :, i_time]), torch.max(dudx[i_sample, :, :, i_time])
        # gt_u.set_clim(cbar_min, cbar_max)
        # gt_v.set_clim(-cbar_max, -cbar_min)
        ax[0].set_title("dudx")
        ax[1].set_title("-dvdy")
        ax[2].set_title("divergence")
        plt.tight_layout()
        plt.show()

    i_plot_velocity = 0
    if i_plot_velocity:
        plot_frames = [0, 7, 14, 21, 28]
        plt.ioff()
        plt.rcParams["font.family"] = "Times New Roman"
        interp_coeff, interp_res = 'none', 'spline16'
        fig, ax = plt.subplots(2, len(plot_frames), figsize=(6, 6), subplot_kw={'xticks': [], 'yticks': []})
        for ii in range(len(plot_frames)):
            gt_u = ax[0, ii].imshow(u_orig[0, :, :, plot_frames[ii]], interpolation=interp_coeff)
            # fig.colorbar(gt_u, ax=ax[0, ii], shrink=0.4)
            gt_v = ax[1, ii].imshow(v_orig[0, :, :, plot_frames[ii]], interpolation=interp_coeff)
            # fig.colorbar(gt_v, ax=ax[1, ii], shrink=0.4)
        plt.tight_layout()
        plt.show()

    div_norms = torch.norm(i_div_free.reshape(batchsize, -1, 1), 2, 1)
    u_norms = torch.norm(u_orig.reshape(batchsize, -1, 1), 2, 1)
    v_norms = torch.norm(v_orig.reshape(batchsize, -1, 1), 2, 1)
    l2_rel_u = torch.sum((div_norms / u_norms).mean(-1)) / batchsize
    l2_rel_v = torch.sum((div_norms / v_norms).mean(-1)) / batchsize

    l2_norm_sum = []
    for i in range(batchsize):
        tmp = torch.sqrt(torch.sum((i_div_free[i, ...].reshape(-1)) ** 2))
        l2_norm_sum.append(tmp)
    return torch.sum(div_norms).item()


# function to check if a 3D field is divergence free
def div_free_3d_check(hu, hv, h):
    h_orig = h.clone()
    hu_orig = hu.clone()
    hv_orig = hv.clone()

    batchsize, size_x, size_y, size_z = h.shape[0], h.shape[1], h.shape[2], h.shape[3]
    hu = hu.permute(0, 4, 2, 3, 1)
    hv = hv.permute(0, 4, 1, 3, 2)
    h = h.permute(0, 4, 1, 2, 3)

    # experiment = 'speedyweather'
    experiment = 'rdb'
    # get grids
    if experiment == 'rdb':
        gridx = torch.linspace(-2.5, 2.5, size_x)
        gridy = torch.linspace(-2.5, 2.5, size_y)
        gridz = torch.linspace(0, 1, size_z)
        domain_size_x = 5
        domain_size_y = 5
        domain_size_z = 1
    elif experiment == 'speedyweather':
        sample_rate = 1
        radius_earth = 6.371e6
        # delta_t = 6 * 3600 / radius_earth
        delta_t = 15 * 60 / radius_earth
        # delta_t = 6 * 3600 / 6.371e6 * 2
        # delta_t = delta_t / 7
        # gridx = torch.arange(0, 2.0 * torch.pi, 2.0 * torch.pi / size_x)
        # gridy = torch.linspace(-1.5458759662006802, 1.5458759662006802, size_y)
        gridx = lon
        gridy = lat
        t4_speedyweather = delta_t
        # gridz = torch.linspace(t4_speedyweather, t4_speedyweather + (size_z - 1) * delta_t * sample_rate, size_z)
        gridz = torch.linspace(0, (size_z - 1) * delta_t * sample_rate, size_z)
        domain_size_x = torch.max(gridx) - torch.min(gridx)
        domain_size_y = torch.max(gridy) - torch.min(gridy)
        domain_size_z = (size_z - 1) * delta_t * sample_rate
    else:
        gridx = torch.linspace(0, 1, size_x)
        gridy = torch.linspace(0, 1, size_y)
        gridz = torch.linspace(0, 1, size_z)
        domain_size_x = domain_size_y = domain_size_z = 1

    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    grid = torch.cat((gridx, gridy, gridz), dim=-1)

    n_pts_x, n_pts_y, n_pts_z = h.shape[-3], h.shape[-2], h.shape[-1]
    fc_d, fc_C, fc_filter = 3, 25, False
    fc_pad = fc(fc_d=fc_d, fc_C=fc_C)
    h, fc_prd_h = fc_pad(h, domain_size=domain_size_z)
    hu, fc_prd_hu = fc_pad(hu, domain_size=domain_size_x)
    hu = hu.permute(0, 1, 4, 2, 3)
    hv, fc_prd_hv = fc_pad(hv, domain_size=domain_size_y)
    hv = hv.permute(0, 1, 2, 4, 3)

    # h
    fc_npoints_total = n_pts_z + fc_C
    kx = torch.fft.fftfreq(n_pts_x).to(h.device) * 2 * torch.pi * n_pts_x / (
            grid[0, -1, 0, 0, 0] - grid[0, 0, 0, 0, 0])
    ky = torch.fft.fftfreq(n_pts_y).to(h.device) * 2 * torch.pi * n_pts_y / (
            grid[0, 0, -1, 0, 1] - grid[0, 0, 0, 0, 1])
    kz = torch.fft.fftfreq(fc_npoints_total).to(h.device) * 2 * torch.pi * fc_npoints_total / fc_prd_h
    # get reciprocal grids (wave number axes)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    h_ft = torch.fft.fftn(h, dim=[-3, -2, -1])
    dhdz = torch.real(torch.fft.ifftn(1j * Kz * h_ft, s=(h.size(-3), h.size(-2), h.size(-1)))
                      )[..., :-fc_C].permute(0, 2, 3, 4, 1)

    # hu
    fc_npoints_total = n_pts_x + fc_C
    kx = torch.fft.fftfreq(fc_npoints_total).to(h.device) * 2 * torch.pi * fc_npoints_total / fc_prd_hu
    ky = torch.fft.fftfreq(n_pts_y).to(h.device) * 2 * torch.pi * n_pts_y / (
            grid[0, 0, -1, 0, 1] - grid[0, 0, 0, 0, 1])
    kz = torch.fft.fftfreq(n_pts_z).to(h.device) * 2 * torch.pi * n_pts_z / (
            grid[0, 0, 0, -1, 2] - grid[0, 0, 0, 0, 2])
    # get reciprocal grids (wave number axes)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    hu_ft = torch.fft.fftn(hu, dim=[-3, -2, -1])
    dhudx = torch.real(torch.fft.ifftn(1j * Kx * hu_ft, s=(hu.size(-3), hu.size(-2), hu.size(-1)))
                       )[:, :, :-fc_C, :, :].permute(0, 2, 3, 4, 1)

    # hv
    fc_npoints_total = n_pts_y + fc_C
    kx = torch.fft.fftfreq(n_pts_x).to(h.device) * 2 * torch.pi * n_pts_x / (
            grid[0, -1, 0, 0, 0] - grid[0, 0, 0, 0, 0])
    ky = torch.fft.fftfreq(fc_npoints_total).to(h.device) * 2 * torch.pi * fc_npoints_total / fc_prd_hv
    kz = torch.fft.fftfreq(n_pts_z).to(h.device) * 2 * torch.pi * n_pts_z / (
            grid[0, 0, 0, -1, 2] - grid[0, 0, 0, 0, 2])
    # get reciprocal grids (wave number axes)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    hv_ft = torch.fft.fftn(hv, dim=[-3, -2, -1])
    dhvdy = torch.real(torch.fft.ifftn(1j * Ky * hv_ft, s=(hv.size(-3), hv.size(-2), hv.size(-1)))
                       )[:, :, :, :-fc_C, :].permute(0, 2, 3, 4, 1)

    scale_factor = 1.0

    # i_div_free = dhudx - dhvdy + dhdz if experiment == 'speedyweather' else dhudx + dhvdy + dhdz
    # deleteme = dhudx - dhvdy if experiment == 'speedyweather' else dhudx + dhvdy
    i_div_free = dhudx - dhvdy + scale_factor * dhdz if experiment == 'speedyweather' else dhudx + dhvdy + dhdz
    deleteme = dhudx - dhvdy if experiment == 'speedyweather' else dhudx + dhvdy

    # divergence using central difference
    x_ref = hu_orig.squeeze().numpy()[0, ...]
    y_ref = hv_orig.squeeze().numpy()[0, ...]
    z_ref = h_orig.squeeze().numpy()[0, ...]
    dx = (grid[0, 1, 0, 0, 0] - grid[0, 0, 0, 0, 0]).item()
    dy = (grid[0, 0, 1, 0, 1] - grid[0, 0, 0, 0, 1]).item()
    dz = (grid[0, 0, 0, 1, 2] - grid[0, 0, 0, 0, 2]).item()
    i_div_free_ref = np.gradient(x_ref, dx, dy, dz)[0] - np.gradient(y_ref, dx, dy, dz)[1] + scale_factor*np.gradient(z_ref, dx, dy, dz)[2] if experiment == 'speedyweather' else np.gradient(x_ref, dx, dy, dz)[0] + np.gradient(y_ref, dx, dy, dz)[1] + scale_factor*np.gradient(z_ref, dx, dy, dz)[2]
    deleteme_ref = np.gradient(x_ref, dx, dy, dz)[0] - np.gradient(y_ref, dx, dy, dz)[1] if experiment == 'speedyweather' else np.gradient(x_ref, dx, dy, dz)[0] + np.gradient(y_ref, dx, dy, dz)[1]
    dhdz_ref = np.gradient(z_ref, dx, dy, dz)[2]

    aaa=pearsonr(deleteme_ref.reshape(-1), dhdz_ref.reshape(-1))

    i_plot_data = True
    if i_plot_data:
        i_time = -1
        plt.ioff()
        plt.rcParams["font.family"] = "Times New Roman"
        interp_coeff, interp_res = 'none', 'spline16'
        fig, ax = plt.subplots(2, 3, figsize=(6, 6), subplot_kw={'xticks': [], 'yticks': []})
        # gt_u = ax[0, 0].imshow(dhdz[0, :, :, i_time, 0], origin='lower', aspect='auto', interpolation=interp_coeff)
        gt_u = ax[0, 0].imshow(dhdz[0, :, :, i_time, 0], interpolation=interp_coeff)
        fig.colorbar(gt_u, ax=ax[0, 0], shrink=0.4)
        gt_v = ax[0, 1].imshow(-deleteme[0, :, :, i_time, 0], interpolation=interp_coeff)
        fig.colorbar(gt_v, ax=ax[0, 1], shrink=0.4)
        gt_div = ax[0, 2].imshow(i_div_free[0, :, :, i_time, 0], interpolation=interp_coeff)
        fig.colorbar(gt_div, ax=ax[0, 2], shrink=0.4)

        gt_u_ref = ax[1, 0].imshow(dhdz_ref[:, :, i_time], interpolation=interp_coeff)
        fig.colorbar(gt_u_ref, ax=ax[1, 0], shrink=0.4)
        gt_v_ref = ax[1, 1].imshow(-deleteme_ref[:, :, i_time], interpolation=interp_coeff)
        fig.colorbar(gt_v_ref, ax=ax[1, 1], shrink=0.4)
        gt_div_ref = ax[1, 2].imshow(i_div_free_ref[:, :, i_time], interpolation=interp_coeff)
        fig.colorbar(gt_div_ref, ax=ax[1, 2], shrink=0.4)

        # cbar_min, cbar_max = torch.min(dhdz[0, :, :, i_time, 0]), torch.max(dhdz[0, :, :, i_time, 0])
        cbar_min, cbar_max = np.min(dhdz_ref[:, :, i_time]), np.max(dhdz_ref[:, :, i_time])
        gt_u.set_clim(cbar_min, cbar_max)
        gt_u_ref.set_clim(cbar_min, cbar_max)
        cbar_min, cbar_max = np.min(-deleteme_ref[:, :, i_time]), np.max(-deleteme_ref[:, :, i_time])
        gt_v.set_clim(cbar_min, cbar_max)
        gt_v_ref.set_clim(cbar_min, cbar_max)
        cbar_min, cbar_max = np.min(i_div_free_ref[:, :, i_time]), np.max(i_div_free_ref[:, :, i_time])
        gt_div.set_clim(cbar_min, cbar_max)
        gt_div_ref.set_clim(cbar_min, cbar_max)
        ax[0, 0].set_title("dhdt")
        ax[0, 1].set_title("-(dhudx + dhvdy)")
        ax[0, 2].set_title("divergence")

        ax[0, 0].set_ylabel("FFT derivative")
        ax[1, 0].set_ylabel("numpy gradient")
        plt.tight_layout()
        plt.show()

    div_norms = torch.norm(i_div_free.reshape(batchsize, -1, 1), 2, 1)
    h_norms = torch.norm(h_orig.reshape(batchsize, -1, 1), 2, 1)
    hu_norms = torch.norm(hu_orig.reshape(batchsize, -1, 1), 2, 1)
    hv_norms = torch.norm(hv_orig.reshape(batchsize, -1, 1), 2, 1)
    l2_rel_h = torch.sum((div_norms / h_norms).mean(-1)) / batchsize
    l2_rel_hu = torch.sum((div_norms / hu_norms).mean(-1)) / batchsize
    l2_rel_hv = torch.sum((div_norms / hv_norms).mean(-1)) / batchsize

    l2_norm_sum = []
    for i in range(batchsize):
        tmp = torch.sqrt(torch.sum((i_div_free[i, ...].reshape(-1)) ** 2))
        l2_norm_sum.append(tmp)
    return i_div_free


# function to check if a 3D field is divergence free
def div_free_3d_check_speedyweather(hu, hv, h, lon, lat):
    h_orig = h.clone()
    hu_orig = hu.clone()
    hv_orig = hv.clone()

    batchsize, size_x, size_y, size_z = h.shape[0], h.shape[1], h.shape[2], h.shape[3]
    hu = hu.permute(0, 4, 2, 3, 1)
    hv = hv.permute(0, 4, 1, 3, 2)
    h = h.permute(0, 4, 1, 2, 3)

    experiment = 'speedyweather'
    # get grids
    if experiment == 'rdb':
        gridx = torch.linspace(-2.5, 2.5, size_x)
        gridy = torch.linspace(-2.5, 2.5, size_y)
        gridz = torch.linspace(0, 1, size_z)
        domain_size_x = 5
        domain_size_y = 5
        domain_size_z = 1
    elif experiment == 'speedyweather':
        sample_rate = 1
        radius_earth = 6.371e6
        # delta_t = 6 * 3600 / radius_earth
        delta_t = 15 * 60 / radius_earth
        # delta_t *= 174.0
        # delta_t = 6 * 3600 / 6.371e6 * 2
        # delta_t = delta_t / 7
        # gridx = torch.arange(0, 2.0 * torch.pi, 2.0 * torch.pi / size_x)
        # gridy = torch.linspace(-1.5458759662006802, 1.5458759662006802, size_y)
        gridx = lon
        gridy = lat
        t4_speedyweather = delta_t
        # gridz = torch.linspace(t4_speedyweather, t4_speedyweather + (size_z - 1) * delta_t * sample_rate, size_z)
        gridz = torch.linspace(0, (size_z - 1) * delta_t * sample_rate, size_z)
        domain_size_x = torch.max(gridx) - torch.min(gridx)
        domain_size_y = torch.max(gridy) - torch.min(gridy)
        domain_size_z = (size_z - 1) * delta_t * sample_rate
    else:
        gridx = torch.linspace(0, 1, size_x)
        gridy = torch.linspace(0, 1, size_y)
        gridz = torch.linspace(0, 1, size_z)
        domain_size_x = domain_size_y = domain_size_z = 1

    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    grid = torch.cat((gridx, gridy, gridz), dim=-1)

    n_pts_x, n_pts_y, n_pts_z = h.shape[-3], h.shape[-2], h.shape[-1]
    fc_d, fc_C, fc_filter = 3, 25, False
    fc_pad = fc(fc_d=fc_d, fc_C=fc_C)
    h, fc_prd_h = fc_pad(h, domain_size=domain_size_z)
    hu, fc_prd_hu = fc_pad(hu, domain_size=domain_size_x)
    hu = hu.permute(0, 1, 4, 2, 3)
    hv, fc_prd_hv = fc_pad(hv, domain_size=domain_size_y)
    hv = hv.permute(0, 1, 2, 4, 3)

    # h
    fc_npoints_total = n_pts_z + fc_C
    kx = torch.fft.fftfreq(n_pts_x).to(h.device) * 2 * torch.pi * n_pts_x / (
            grid[0, -1, 0, 0, 0] - grid[0, 0, 0, 0, 0])
    ky = torch.fft.fftfreq(n_pts_y).to(h.device) * 2 * torch.pi * n_pts_y / (
            grid[0, 0, -1, 0, 1] - grid[0, 0, 0, 0, 1])
    kz = torch.fft.fftfreq(fc_npoints_total).to(h.device) * 2 * torch.pi * fc_npoints_total / fc_prd_h
    # get reciprocal grids (wave number axes)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    h_ft = torch.fft.fftn(h, dim=[-3, -2, -1])
    dhdz = torch.real(torch.fft.ifftn(1j * Kz * h_ft, s=(h.size(-3), h.size(-2), h.size(-1)))
                      )[..., :-fc_C].permute(0, 2, 3, 4, 1)

    # hu
    fc_npoints_total = n_pts_x + fc_C
    kx = torch.fft.fftfreq(fc_npoints_total).to(h.device) * 2 * torch.pi * fc_npoints_total / fc_prd_hu
    ky = torch.fft.fftfreq(n_pts_y).to(h.device) * 2 * torch.pi * n_pts_y / (
            grid[0, 0, -1, 0, 1] - grid[0, 0, 0, 0, 1])
    kz = torch.fft.fftfreq(n_pts_z).to(h.device) * 2 * torch.pi * n_pts_z / (
            grid[0, 0, 0, -1, 2] - grid[0, 0, 0, 0, 2])
    # get reciprocal grids (wave number axes)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    hu_ft = torch.fft.fftn(hu, dim=[-3, -2, -1])
    dhudx = torch.real(torch.fft.ifftn(1j * Kx * hu_ft, s=(hu.size(-3), hu.size(-2), hu.size(-1)))
                       )[:, :, :-fc_C, :, :].permute(0, 2, 3, 4, 1)

    # hv
    fc_npoints_total = n_pts_y + fc_C
    kx = torch.fft.fftfreq(n_pts_x).to(h.device) * 2 * torch.pi * n_pts_x / (
            grid[0, -1, 0, 0, 0] - grid[0, 0, 0, 0, 0])
    ky = torch.fft.fftfreq(fc_npoints_total).to(h.device) * 2 * torch.pi * fc_npoints_total / fc_prd_hv
    kz = torch.fft.fftfreq(n_pts_z).to(h.device) * 2 * torch.pi * n_pts_z / (
            grid[0, 0, 0, -1, 2] - grid[0, 0, 0, 0, 2])
    # get reciprocal grids (wave number axes)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    hv_ft = torch.fft.fftn(hv, dim=[-3, -2, -1])
    dhvdy = torch.real(torch.fft.ifftn(1j * Ky * hv_ft, s=(hv.size(-3), hv.size(-2), hv.size(-1)))
                       )[:, :, :, :-fc_C, :].permute(0, 2, 3, 4, 1)

    # scale_factor = 1.35
    scale_factor = 1.0

    # i_div_free = dhudx - dhvdy + dhdz if experiment == 'speedyweather' else dhudx + dhvdy + dhdz
    # deleteme = dhudx - dhvdy if experiment == 'speedyweather' else dhudx + dhvdy
    i_div_free = dhudx - dhvdy + scale_factor * dhdz if experiment == 'speedyweather' else dhudx + dhvdy + dhdz
    deleteme = dhudx - dhvdy if experiment == 'speedyweather' else dhudx + dhvdy

    # divergence using central difference
    x_ref = hu_orig.squeeze().numpy()
    y_ref = hv_orig.squeeze().numpy()
    z_ref = h_orig.squeeze().numpy()
    dx = (lon[1] - lon[0]).item()
    dy = (lat[1] - lat[0]).item()
    dz = delta_t * sample_rate
    i_div_free_ref = np.gradient(x_ref, dx, dy, dz)[0] - np.gradient(y_ref, dx, dy, dz)[1] + scale_factor*np.gradient(z_ref, dx, dy, dz)[2] if experiment == 'speedyweather' else np.gradient(x_ref, dx, dy, dz)[0] + np.gradient(y_ref, dx, dy, dz)[1] + scale_factor*np.gradient(z_ref, dx, dy, dz)[2]
    deleteme_ref = np.gradient(x_ref, dx, dy, dz)[0] - np.gradient(y_ref, dx, dy, dz)[1] if experiment == 'speedyweather' else np.gradient(x_ref, dx, dy, dz)[0] + np.gradient(y_ref, dx, dy, dz)[1]
    dhdz_ref = np.gradient(z_ref, dx, dy, dz)[2]

    aaa=pearsonr(deleteme_ref.reshape(-1), dhdz_ref.reshape(-1))

    i_plot_data = True
    if i_plot_data:
        i_time = -2
        plt.ioff()
        plt.rcParams["font.family"] = "Times New Roman"
        interp_coeff, interp_res = 'none', 'spline16'
        fig, ax = plt.subplots(2, 3, figsize=(6, 6), subplot_kw={'xticks': [], 'yticks': []})
        # gt_u = ax[0, 0].imshow(dhdz[0, :, :, i_time, 0], origin='lower', aspect='auto', interpolation=interp_coeff)
        gt_u = ax[0, 0].imshow(dhdz[0, :, :, i_time, 0].T, interpolation=interp_coeff)
        fig.colorbar(gt_u, ax=ax[0, 0], shrink=0.4)
        gt_v = ax[0, 1].imshow(-deleteme[0, :, :, i_time, 0].T, interpolation=interp_coeff)
        fig.colorbar(gt_v, ax=ax[0, 1], shrink=0.4)
        gt_div = ax[0, 2].imshow(i_div_free[0, :, :, i_time, 0].T, interpolation=interp_coeff)
        fig.colorbar(gt_div, ax=ax[0, 2], shrink=0.4)
        # gt_u.set_clim(-1, 1)
        # gt_v.set_clim(-1, 1)
        # gt_div.set_clim(-1, 1)

        gt_u_ref = ax[1, 0].imshow(dhdz_ref[:, :, i_time].T, interpolation=interp_coeff)
        fig.colorbar(gt_u_ref, ax=ax[1, 0], shrink=0.4)
        gt_v_ref = ax[1, 1].imshow(-deleteme_ref[:, :, i_time].T, interpolation=interp_coeff)
        fig.colorbar(gt_v_ref, ax=ax[1, 1], shrink=0.4)
        gt_div_ref = ax[1, 2].imshow(i_div_free_ref[:, :, i_time].T, interpolation=interp_coeff)
        fig.colorbar(gt_div_ref, ax=ax[1, 2], shrink=0.4)
        cbar_min, cbar_max = torch.min(dhdz[0, :, :, i_time, 0]), torch.max(dhdz[0, :, :, i_time, 0])
        gt_u.set_clim(cbar_min, cbar_max)
        gt_u_ref.set_clim(cbar_min, cbar_max)

        cbar_min, cbar_max = np.min(-deleteme_ref[:, :, i_time]), np.max(-deleteme_ref[:, :, i_time])
        gt_v.set_clim(cbar_min, cbar_max)
        gt_v_ref.set_clim(cbar_min, cbar_max)

        cbar_min, cbar_max = np.min(i_div_free_ref[:, :, i_time]), np.max(i_div_free_ref[:, :, i_time])
        gt_div.set_clim(cbar_min, cbar_max)
        gt_div_ref.set_clim(cbar_min, cbar_max)

        # gt_u.set_clim(-1, 1)
        # gt_v.set_clim(-1, 1)
        # gt_div.set_clim(-1, 1)

        ax[0, 0].set_title("dhdt")
        ax[0, 1].set_title("-(dhudx + dhvdy)")
        ax[0, 2].set_title("divergence")

        ax[0, 0].set_ylabel("FFT derivative")
        ax[1, 0].set_ylabel("numpy gradient")
        plt.tight_layout()
        plt.show()

    div_norms = torch.norm(i_div_free.reshape(batchsize, -1, 1), 2, 1)
    h_norms = torch.norm(h_orig.reshape(batchsize, -1, 1), 2, 1)
    hu_norms = torch.norm(hu_orig.reshape(batchsize, -1, 1), 2, 1)
    hv_norms = torch.norm(hv_orig.reshape(batchsize, -1, 1), 2, 1)
    l2_rel_h = torch.sum((div_norms / h_norms).mean(-1)) / batchsize
    l2_rel_hu = torch.sum((div_norms / hu_norms).mean(-1)) / batchsize
    l2_rel_hv = torch.sum((div_norms / hv_norms).mean(-1)) / batchsize

    l2_norm_sum = []
    for i in range(batchsize):
        tmp = torch.sqrt(torch.sum((i_div_free[i, ...].reshape(-1)) ** 2))
        l2_norm_sum.append(tmp)
    return i_div_free


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size() + (2,) if p.is_complex() else p.size()))
    return c
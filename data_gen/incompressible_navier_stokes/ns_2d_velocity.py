import torch
import math
from random_fields import GaussianRF
from timeit import default_timer
import scipy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt


# w0: initial vorticity
# f: forcing term
# visc: viscosity (1/Re)
# T: final time
# delta_t: internal time-step for solve (descrease if blow-up)
# record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    # Grid size - must be power of 2
    N = w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N / 2.0)

    # Number of steps to final time
    steps = math.floor(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)

    # Forcing to Fourier space
    f_h = torch.fft.rfft2(f)

    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Record solution every this number of steps
    record_time = math.floor(steps / record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                     torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0
    # Dealiasing mask
    dealias = torch.unsqueeze(
        torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max, torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)

    # Saving solution and time
    sol_u = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_v = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in range(steps):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        # Velocity field in x-direction = psi_y
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        # Velocity field in y-direction = -psi_x
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        # Partial x of vorticity
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        # Partial y of vorticity
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q * w_x + v * w_y)

        # Dealias
        F_h = dealias * F_h

        # Crank-Nicolson update
        w_h = (-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h) / (
                    1.0 + 0.5 * delta_t * visc * lap)

        # Update real time (used only for recording)
        t += delta_t

        if (j + 1) % record_time == 0:
            # Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))

            # Record solution and time
            sol_u[..., c] = q
            sol_v[..., c] = v
            sol_t[c] = t

            c += 1

    return sol_u, sol_v, sol_t


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print(f'>> Device being used: {device}')
else:
    print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

# Resolution
s = 256

# Number of solutions to generate
N = 100
# N = 1

# Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s + 1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t, indexing='ij')
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

# Number of snapshots from solution
record_steps = 422
output_per_n_steps = 2000

# Inputs
a = torch.zeros(N, s, s)
# Solutions
u = torch.zeros(N, s, s, record_steps)
v = torch.zeros(N, s, s, record_steps)

# Solve equations in batches (order of magnitude speed-up)

# Batch size
bsize = 20
# bsize = 1

c = 0
t0 = default_timer()

dt = 1e-4
visc = 1e-4
t_final = record_steps * dt * output_per_n_steps
# t_final = 50
steps = math.floor(t_final / dt)
record_time = math.floor(steps / record_steps)
print(f'>> Record solution every {record_time} number of steps')

for j in range(N // bsize):
    # Sample random feilds
    w0 = GRF.sample(bsize)

    # Solve NS
    sol_u, sol_v, sol_t = navier_stokes_2d(w0, f, visc, t_final, dt, record_steps)

    a[c:(c + bsize), ...] = w0
    u[c:(c + bsize), ...] = sol_u
    v[c:(c + bsize), ...] = sol_v

    c += bsize
    t1 = default_timer()
    print(j, c, t1 - t0)

i_plot_velocity = 0
if i_plot_velocity:
    plt.ioff()
    plt.rcParams["font.family"] = "Times New Roman"
    interp_coeff, interp_res = 'none', 'spline16'
    fig, ax = plt.subplots(2, 5, figsize=(6, 6), subplot_kw={'xticks': [], 'yticks': []})
    plot_frames = [0, 30, 60, 90, 120]
    for ii in range(5):
        gt_u = ax[0, ii].imshow(u[0, :, :, plot_frames[ii]], interpolation=interp_coeff)
        fig.colorbar(gt_u, ax=ax[0, ii], shrink=0.4)
        gt_v = ax[1, ii].imshow(v[0, :, :, plot_frames[ii]], interpolation=interp_coeff)
        fig.colorbar(gt_v, ax=ax[1, ii], shrink=0.4)
    plt.tight_layout()
    plt.show()

print(f'>> velocity u shape: {u.shape}')
# scipy.io.savemat('ns_data_zongyili.mat', mdict={'u': u.cpu().numpy(), 'v': v.cpu().numpy(), 't': sol_t.cpu().numpy()})
with h5py.File('./ns_data_samplefreq%d.h5' % output_per_n_steps, 'w') as fs:
    fs.create_dataset('u', data=u.cpu().numpy(), dtype=np.float32)
    fs.create_dataset('v', data=v.cpu().numpy(), dtype=np.float32)
    # fs.create_dataset('t', data=sol_t.cpu().numpy(), dtype=np.float32)

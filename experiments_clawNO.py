"""
This is a modified version of fourier_2d_time.py from https://github.com/zongyi-li/fourier_neural_operator
"""
import datetime
import os
import random
from scipy.interpolate import griddata

# os.environ["CUDA_VISIBLE_DEVICES"]="7" # TODO: for debugging
from models.FNO import FNO2d, FNO3d
from models.clawFNO import clawFNO2d, clawFNO3d
from models.GFNO import GFNO2d, GFNO3d
from models.clawGFNO import clawGFNO2d, clawGFNO3d
# from models.GFNO_steerable import GFNO2d_steer
# from models.Unet import Unet_Rot, Unet_Rot_M, Unet_Rot_3D
from models.unet_orig import UNet1d, UNet2d, UNet3d
from models.Ghybrid import Ghybrid2d
from models.radialNO import radialNO2d, radialNO3d
from models.GCNN import GCNN2d, GCNN3d

from utils import pde_data, LpLoss, eq_check_rt, eq_check_rf, div_free_2d_check, div_free_3d_check, count_params, \
    div_free_3d_check_speedyweather

import scipy
import numpy as np
from timeit import default_timer
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import h5py
import xarray as xr
from tqdm import tqdm

torch.set_num_threads(1)


def get_eval_pred(model, x, strategy, T, times):
    if strategy == "oneshot":
        pred = model(x)
    else:
        for t in range(T):
            t1 = default_timer()
            im = model(x)
            times.append(default_timer() - t1)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -2)
            if strategy == "markov":
                x = im
            else:
                x = torch.cat((x[..., 1:, :], im), dim=-2)

    return pred


def main(args):
    assert args.model_type in ("FNO2d", "FNO2d_aug",
                               "FNO3d", "FNO3d_aug",
                               "clawFNO2d", "clawFNO3d",
                               "GCNN2d_p4", "GCNN2d_p4m",
                               "GCNN3d_p4", "GCNN3d_p4m",
                               "GFNO2d_p4", "GFNO2d_p4m",
                               "GFNO2d_p4_steer", "GFNO2d_p4m_steer",
                               "GFNO3d_p4", "GFNO3d_p4m",
                               "clawGFNO2d_p4", "clawGFNO2d_p4m",
                               "clawGFNO3d_p4", "clawGFNO3d_p4m",
                               "Ghybrid2d_p4", "Ghybrid2d_p4m",
                               "radialNO2d_p4", "radialNO2d_p4m",
                               "radialNO3d_p4", "radialNO3d_p4m",
                               "Unet_Rot2d", "Unet_Rot_M2d", "Unet_Rot_3D",
                               "UNet2d", "UNet3d"), f"Invalid model type {args.model_type}"
    assert args.strategy in ["teacher_forcing", "markov", "recurrent", "oneshot"], "Invalid training strategy"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    data_aug = "aug" in args.model_type

    TRAIN_PATH = args.data_path

    # FNO data specs
    S = Sx = Sy = 64  # spatial res
    S_super = 4 * S  # super spatial res
    T_in = 10  # number of input times
    T = args.T
    T_super = 4 * T  # prediction temporal super res
    d = 2  # spatial res
    num_channels = 1

    # adjust data specs based on model type and data path
    threeD = args.model_type in ("FNO3d", "FNO3d_aug",
                                 "GCNN3d_p4", "GCNN3d_p4m",
                                 "GFNO3d_p4", "GFNO3d_p4m",
                                 "radialNO3d_p4", "radialNO3d_p4m",
                                 "clawFNO3d", "Unet_Rot_3D", "UNet3d",
                                 "clawGFNO3d_p4", "clawGFNO3d_p4m")
    extension = TRAIN_PATH.split(".")[-1]
    # swe = os.path.split(TRAIN_PATH)[-1] == "ShallowWater2D"
    swe = TRAIN_PATH.split("/")[2][:2] == "sw"
    ns_zli = TRAIN_PATH.split("/")[2][-6:] == "zli.h5"
    ns = TRAIN_PATH.split("/")[2][:2] == "ns"
    # ns = True
    # rdb = TRAIN_PATH.split(os.path.sep)[-1][:6] == "2D_rdb"
    rdb = TRAIN_PATH.split('/')[-1][:6] == "2D_rdb"
    grid_type = "symmetric"
    if args.grid:
        grid_type = args.grid
        assert grid_type in ['symmetric', 'cartesian', 'None']

    if rdb:
        assert T == 24, "T should be 24 for rdb"
        T_in = 1
        S = Sx = Sy = 32
        # num_channels = 1
        num_channels = 3  # (h, u, v)
        S_super = 128
        T_super = 96
    elif swe:
        assert not args.super, "Super-resolution not supported for pdearena"
        # assert T == 10, "T should be 10 for swe"
        T_in = 1
        Sy, Sx = 95, 192
        num_channels = 3
        grid_type = "cartesian"
    elif ns:
        assert T == 20, "T should be 20 for ns"
        T_in = 10
        S = Sx = Sy = 64
        num_channels = 2  # (u, v)
        S_super = 4 * S  # super spatial res
        T_super = 4 * T  # prediction temporal super res
    spatial_dims = range(1, d + 1)

    if args.strategy == "oneshot":
        assert threeD, "oneshot strategy only for 3d models"

    if threeD:
        assert args.strategy == "oneshot", "threeD models use oneshot strategy"
        # assert args.modes <= 8, "modes for 3d models should be leq 8"

    ntrain = args.ntrain  # 1000
    nvalid = args.nvalid
    ntest = args.ntest  # 200

    time_modes = None
    time = args.strategy == "oneshot"  # perform convolutions in space-time
    if args.time_pad:
        print(f'*************************** Note: time padding is turned on!! ***************************')
    if time and not args.time_pad:
        time_modes = 5 if swe else 6  # 6 is based on T=10
    elif time and swe:
        time_modes = 8

    modes = args.modes
    width = args.width
    n_layer = args.depth
    batch_size = args.batch_size

    epochs = args.epochs  # 500
    learning_rate = args.learning_rate
    scheduler_step = args.step_size
    scheduler_gamma = args.gamma  # for step scheduler

    initial_step = 1 if args.strategy == "markov" else T_in

    # root = args.results_path + f"/{'_'.join(str(datetime.datetime.now()).split())}"
    # if args.suffix:
    #     root += "_" + args.suffix
    # root = root.replace(':', '_')

    root = args.results_path + 'grid_%s_n%d/seed%d_lr%e_wd%e' % (
        grid_type, ntrain, args.seed, args.learning_rate, args.lmbda)
    if not os.path.exists(root):
        os.makedirs(root)
    path_model = os.path.join(root, 'model.pt')
    writer = SummaryWriter(root)

    ################################################################
    # Model init
    ################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

    if args.model_type in ["FNO2d", "FNO2d_aug"]:
        model = FNO2d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, width=width,
                      grid_type=grid_type).to(device)
    elif args.model_type in ["FNO3d", "FNO3d_aug"]:
        modes3 = time_modes if time_modes else modes
        model = FNO3d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, modes3=modes3,
                      width=width, grid_type=grid_type, time=time, time_pad=args.time_pad).to(device)
    elif args.model_type in ["clawFNO3d"]:
        modes3 = time_modes if time_modes else modes
        model = clawFNO3d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes,
                          modes3=modes3, width=width, grid_type=grid_type, time=time, time_pad=args.time_pad).to(device)
    elif args.model_type in ["clawFNO2d"]:
        model = clawFNO2d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, width=width,
                          grid_type=grid_type).to(device)
    elif "GCNN2d" in args.model_type:
        reflection = "p4m" in args.model_type
        model = GCNN2d(num_channels=num_channels, initial_step=initial_step, width=width, reflection=reflection).to(
            device)
    elif "GCNN3d" in args.model_type:
        reflection = "p4m" in args.model_type
        model = GCNN3d(num_channels=num_channels, initial_step=initial_step, width=width, reflection=reflection).to(
            device)
    elif "clawGFNO2d" in args.model_type:
        reflection = "p4m" in args.model_type
        model = clawGFNO2d(num_channels=num_channels, initial_step=initial_step, modes=modes, width=width,
                           reflection=reflection, grid_type=grid_type).to(device)
    elif "GFNO2d" in args.model_type and "steer" in args.model_type:
        reflection = "p4m" in args.model_type
        model = GFNO2d_steer(num_channels=num_channels, initial_step=initial_step, input_size=S, modes=modes,
                             width=width,
                             reflection=reflection).to(device)
    elif "GFNO2d" in args.model_type:
        reflection = "p4m" in args.model_type
        model = GFNO2d(num_channels=num_channels, initial_step=initial_step, modes=modes, width=width,
                       reflection=reflection, grid_type=grid_type).to(device)
    elif "clawGFNO3d" in args.model_type:
        reflection = "p4m" in args.model_type
        model = clawGFNO3d(num_channels=num_channels, initial_step=initial_step, modes=modes, time_modes=time_modes,
                           width=width, reflection=reflection, grid_type=grid_type, time_pad=args.time_pad).to(device)
    elif "GFNO3d" in args.model_type:
        reflection = "p4m" in args.model_type
        model = GFNO3d(num_channels=num_channels, initial_step=initial_step, modes=modes, time_modes=time_modes,
                       width=width, reflection=reflection, grid_type=grid_type, time_pad=args.time_pad).to(device)
    elif "Ghybrid2d" in args.model_type:
        reflection = "p4m" in args.model_type
        model = Ghybrid2d(num_channels=num_channels, initial_step=initial_step, modes=modes, Gwidth=args.Gwidth,
                          width=width, reflection=reflection, n_equiv=args.n_equiv).to(device)
    elif "radialNO2d" in args.model_type:
        reflection = "p4m" in args.model_type
        model = radialNO2d(num_channels=num_channels, initial_step=initial_step, modes=modes, width=width,
                           reflection=reflection,
                           grid_type=grid_type).to(device)
    elif "radialNO3d" in args.model_type:
        reflection = "p4m" in args.model_type
        model = radialNO3d(num_channels=num_channels, initial_step=initial_step, modes=modes, time_modes=time_modes,
                           width=width, reflection=reflection, grid_type=grid_type, time_pad=args.time_pad).to(device)
    elif args.model_type == "Unet_Rot2d":
        model = Unet_Rot(input_frames=initial_step * num_channels, output_frames=num_channels, kernel_size=3, N=4).to(
            device)
    elif args.model_type == "Unet_Rot_M2d":
        model = Unet_Rot_M(input_frames=initial_step * num_channels, output_frames=num_channels, kernel_size=3, N=4,
                           grid_type=grid_type, width=width).to(device)
    elif args.model_type == "Unet_Rot_3D":
        model = Unet_Rot_3D(input_frames=initial_step * num_channels, output_frames=num_channels, kernel_size=3, N=4,
                            grid_type=grid_type, width=width).to(device)
    elif args.model_type == "UNet2d":
        model = UNet2d(in_channels=initial_step * num_channels, out_channels=num_channels, init_features=width,
                       kernel_size=2).to(device)
    elif args.model_type == "UNet3d":
        model = UNet3d(in_channels=initial_step * num_channels, out_channels=num_channels, init_features=width,
                       kernel_size=2).to(device)
    else:
        raise NotImplementedError("Model not recognized")

    print(f'>> Total number of {args.model_type} model parameters: {count_params(model)}')

    # test model on training-resolution and super-resolution data
    if args.strategy == "oneshot":
        x_shape = [batch_size, Sx, Sy, T, initial_step, num_channels]
        x_shape_super = [1, S_super, S_super, T_super, initial_step, num_channels]
    elif args.strategy == "markov":
        x_shape = [batch_size, Sx, Sy, 1, num_channels]
        x_shape_super = [1, *(S_super,) * d, 1, num_channels]
    else:  # strategy == recurrent or teacher_forcing
        x_shape = [batch_size, Sx, Sy, T_in, num_channels]
        x_shape_super = [1, *(S_super,) * d, T_in, num_channels]

    model.train()
    x = torch.randn(*x_shape).to(device)
    if args.strategy == "recurrent":
        for _ in range(T):
            im = model(x)
            x = torch.cat([x[..., 1:, :], im], dim=-2)
    else:
        model(x)
    # eq_check_rt(model, x, spatial_dims)
    # eq_check_rf(model, x, spatial_dims)
    if args.super:
        model.eval()
        with torch.no_grad():
            x = torch.randn(*x_shape_super).to(device)
            model(x)

    ################################################################
    # load data
    ################################################################
    full_data = None  # for superres
    ns_full_resolution = 0
    ns_zli_read = 0
    if ns_zli and ns_zli_read:  # incompressible NS data in FNO paper
        print(f'>> Reading in zli full incompressible NS data..')
        assert num_channels == 2, "num channels should be 2 for ns data (two velocity components)"
        assert d == 2, "spatial dim should be 2 for ns data"
        sub = 1
        try:
            with h5py.File(TRAIN_PATH, 'r') as f:
                data = np.expand_dims(np.array(f['u'], dtype=np.float32), axis=-1)
                data = np.concatenate((data, np.expand_dims(np.array(f['v'], dtype=np.float32), axis=-1)), axis=-1)
                # tttt = np.array(f['t'], dtype=np.float32)
            # data = np.transpose(data, (0, 2, 3, 1, 4))
        except:
            data = scipy.io.loadmat(os.path.expandvars(TRAIN_PATH))['velocity'].astype(np.float32)
        # remove the 1st timestep which is a random initialization and is not divergence free
        data = data[..., 1:, :]
        sample_rate = 4
        full_data = data[-ntest:, ..., :(T_in + T) * sample_rate, :]
        # data = data[:, ::sample_rate, ::sample_rate, :30, :]
        # data = data[..., :30, :]

        sampler = torch.nn.AvgPool2d(kernel_size=4)
        data = sampler(torch.tensor(data[..., ::sample_rate, :])[..., :T_in + T, :]
                       .reshape(data.shape[0], S_super, S_super, -1).permute(0, 3, 1, 2)) \
            .permute(0, 2, 3, 1).reshape(data.shape[0], Sx, Sy, -1, num_channels).numpy()

        i_save_downsampled_data = 1
        if i_save_downsampled_data:
            fs = h5py.File('./data/ns_downsampled/ns_data4training_zli_samplefreq2e3_dsfreq%d.h5' % sample_rate, 'w')
            fs.create_dataset('velocity', data=data)
            fs.close()

            fs = h5py.File('./data/ns_downsampled/ns_data4superres_zli_samplefreq2e3.h5', 'w')
            fs.create_dataset('velocity', data=full_data)
            fs.close()
    elif ns_full_resolution:  # incompressible NS generated from PDEBench
        print(f'>> Reading in full incompressible NS data..')
        assert num_channels == 2, "num channels should be 2 for ns data (two velocity components)"
        assert d == 2, "spatial dim should be 2 for ns data"
        sub = 1
        try:
            with h5py.File(TRAIN_PATH, 'r') as f:
                data = np.array(f['velocity'], dtype=np.float32)
                # ttt = np.array(f['t'], dtype=np.float32)
            data = np.transpose(data, (0, 2, 3, 1, 4))
        except:
            data = scipy.io.loadmat(os.path.expandvars(TRAIN_PATH))['velocity'].astype(np.float32)
        # remove the 1st timestep which is a random initialization and is not divergence free
        data = data[..., 1:, :]
        sample_rate = 4
        full_data = data[-ntest:, ..., :(T_in + T) * sample_rate, :]
        # data = data[:, ::sample_rate, ::sample_rate, :30, :]
        # data = data[..., :30, :]

        sampler = torch.nn.AvgPool2d(kernel_size=4)
        data = sampler(torch.tensor(data[..., ::sample_rate, :])[..., :T_in + T, :]
                       .reshape(data.shape[0], S_super, S_super, -1).permute(0, 3, 1, 2)) \
            .permute(0, 2, 3, 1).reshape(data.shape[0], Sx, Sy, -1, num_channels).numpy()

        i_save_downsampled_data = 0
        if i_save_downsampled_data:
            fs = h5py.File('./data/ns_downsampled/ns_data4training_zli_dsfreq%d.h5' % sample_rate, 'w')
            fs.create_dataset('velocity', data=data)
            fs.close()

            fs = h5py.File('./data/ns_downsampled/ns_data4superres_zli.h5', 'w')
            fs.create_dataset('velocity', data=full_data)
            fs.close()
    elif ns:  # incompressible NS
        print(f'>> Reading in downsampled incompressible NS data..')
        assert num_channels == 2, "num channels should be 2 for ns data (two velocity components)"
        assert d == 2, "spatial dim should be 2 for ns data"
        sub = 1
        train_path_downsampled = './data/ns_data4training_zli_samplefreq2e3_dsfreq4.h5'
        # train_path_downsampled = './data/ns_data4training_zli_samplefreq1e4_dsfreq4.h5'
        # train_path_downsampled = './data/ns_sim_2d-1.h5'

        # train_path_superres = './data/ns_data4superres_zli.h5'
        try:
            with h5py.File(train_path_downsampled, 'r') as f:
                data = np.array(f['velocity'], dtype=np.float32)
        except:
            data = scipy.io.loadmat(os.path.expandvars(train_path_downsampled))['velocity'].astype(np.float32)

        # try:
        #     with h5py.File(train_path_superres, 'r') as f:
        #         full_data = np.array(f['velocity'], dtype=np.float32)
        # except:
        #     full_data = scipy.io.loadmat(os.path.expandvars(train_path_superres))['velocity'].astype(np.float32)

        # data = np.transpose(data, (0, 2, 3, 1, 4))
        # sample_rate = 4
        # sampler = torch.nn.AvgPool2d(kernel_size=4)
        # data = sampler(torch.tensor(data[..., 1::sample_rate, :])[..., :T_in + T, :]
        #                .reshape(data.shape[0], S_super, S_super, -1).permute(0, 3, 1, 2)) \
        #     .permute(0, 2, 3, 1).reshape(data.shape[0], Sx, Sy, -1, num_channels).numpy()

    elif rdb:  # shallow water equations
        assert num_channels == 3, "num channels should be 3 for uvh shallow water equations"
        # assert num_channels == 1, "num channels should be 3 for uvh shallow water equations"
        assert d == 2, "spatial dim should be 2 for shallow water equations"
        with h5py.File(TRAIN_PATH, 'r') as f:
            data_list = sorted(f.keys())
            print(f'>> The input rdb dataset contains {len(data_list)} samples in total. Reading in..')
            data_h = np.concatenate([np.array(f[key]['data']['h'])[None] for key in data_list]
                                    ).transpose(0, 2, 3, 1, 4)[..., :-1, :]
            data_u = np.concatenate([np.array(f[key]['data']['u'])[None] for key in data_list]
                                    ).transpose(0, 2, 3, 1, 4)[..., :-1, :]
            data_v = np.concatenate([np.array(f[key]['data']['v'])[None] for key in data_list]
                                    ).transpose(0, 2, 3, 1, 4)[..., :-1, :]
            # data = np.concatenate((data_u, data_v, data_h), axis=4)
            data = np.concatenate((data_u * data_h, data_v * data_h, data_h), axis=4)
            # data = data_h
            full_data = data[-ntest:]  # full resolution data for super-resolution test
            sampler = torch.nn.AvgPool2d(kernel_size=4)
            # data = sampler(torch.tensor(data[..., ::4, 0]).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(-1).numpy()
            data = sampler(
                torch.tensor(data[..., ::4, :]).reshape(data.shape[0], S_super, S_super, -1).permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1).reshape(data.shape[0], Sx, Sy, -1, num_channels).numpy()
            # data = sampler(torch.tensor(data[..., ::4, :]).permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2).numpy()
    elif swe:  # swe: # pdearena shallow water equations

        assert num_channels == 2, "num channels should be 2 for shallow water equations"
        assert ntrain + nvalid + ntest <= 5600 + 1120 + 1120, f"Only {5600 + 1120 + 1120} solutions available"
        splits = {"train": ntrain, "valid": nvalid, "test": ntest}
        datas = {}
        for split, n in splits.items():
            if args.verbose: print(f"SWE: loading {split}")
            path = os.path.join(TRAIN_PATH, f"{split}.zarr")
            data = xr.open_zarr(path)
            normstat = torch.load(os.path.join(TRAIN_PATH, "normstats.pt"))

            sample_rate = 8
            VORT_IND = 0
            PRES_IND = 1
            datas[split] = []
            for idx in tqdm(range(n), disable=not args.verbose):
                pres = torch.tensor(data["pres"][idx].to_numpy())
                # pres = (pres - normstat["pres"]["mean"]) / normstat["pres"]["std"]
                h_add = torch.tensor(data["h_add"][idx].to_numpy())
                pres += h_add
                vel_u = torch.tensor(data["u"][idx].to_numpy())
                vel_v = torch.tensor(data["v"][idx].to_numpy())

                pres = pres.unsqueeze(1)
                pres = pres[4::sample_rate]
                vel_u = vel_u[4::sample_rate]
                vel_v = vel_v[4::sample_rate]

                # vort = vort[4::sample_rate]
                vel_h = torch.cat([vel_u * pres, vel_v * pres, pres], dim=1).permute(3, 2, 0, 1).unsqueeze(
                    0)  # Sx,Sy,T,C
                datas[split].append(vel_h)

            datas[split] = torch.cat(datas[split])

        data = torch.cat([datas["train"], datas["valid"], datas["test"]])
    else:
        raise ValueError(f"Extension {extension} not recognized")

    print(f'>> Data reading completed..')
    assert data.shape[-2] >= T + T_in, "not enough time"  # ensure there are enough time steps

    if args.super:
        assert not swe, "Superresolution is not supported for the PDE Arena SWE"
        assert full_data is not None or args.super_path is not None, "missing super dataset"  # ensure theres a dataset for superres

    if not swe:
        data = torch.from_numpy(data)

    assert len(data) >= ntrain + nvalid + ntest, f"not enough data; {len(data)}"

    i_flag_check_div_free_2d = 0
    if i_flag_check_div_free_2d:
        i_div_free = div_free_2d_check(data[:3, ..., 0], data[:3, ..., 1])

    i_flag_check_div_free_3d = 0
    if i_flag_check_div_free_3d:
        if rdb:
            i_div_free = div_free_3d_check(data[..., 0:1], data[..., 1:2], data[..., 2:3])
        elif swe:
            i_div_free = div_free_3d_check_speedyweather(data[..., 0:1], data[..., 1:2], data[..., 2:3], lon, lat)

    train = data[:ntrain]
    assert len(train) == ntrain, "not enough training data"

    test = data[-ntest:]
    test_rt = test.rot90(dims=list(spatial_dims)[:2])
    test_rf = test.flip(dims=(spatial_dims[0],))
    assert len(test) == ntest, "not enough test data"

    valid = data[-(ntest + nvalid):-ntest]
    assert len(valid) == nvalid, "not enough validation data"

    if args.verbose:
        print(f"{args.model_type}: Train/valid/test data shape: ")
        print(train.shape)
        print(valid.shape)
        print(test.shape)

    assert Sx == train.shape[-3], f"Spatial downsampling should give {Sx} grid points"
    assert Sy == train.shape[-4], f"Spatial downsampling should give {Sy} grid points"

    train_data = pde_data(train, strategy=args.strategy, T_in=T_in, T_out=T, std=args.noise_std)
    ntrain = len(train_data)
    valid_data = pde_data(valid, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
    nvalid = len(valid_data)
    test_data = pde_data(test, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
    test_rt_data = pde_data(test_rt, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
    test_rf_data = pde_data(test_rf, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
    ntest = len(test_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    test_rt_loader = torch.utils.data.DataLoader(test_rt_data, batch_size=batch_size, shuffle=False)
    test_rf_loader = torch.utils.data.DataLoader(test_rf_data, batch_size=batch_size, shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################

    complex_ct = sum(par.numel() * (1 + par.is_complex()) for par in model.parameters())
    real_ct = sum(par.numel() for par in model.parameters())
    if args.verbose:
        print(f"{args.model_type}; # Params: complex count {complex_ct}, real count: {real_ct}")
    writer.add_scalar("Parameters/Complex", complex_ct)
    writer.add_scalar("Parameters/Real", real_ct)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.lmbda)
    if args.step:
        print(f'>> Using StepLR instead of CosineAnnealingLR!')
        assert args.step_size is not None, "step_size is None"
        assert scheduler_gamma is not None, "gamma is None"
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=scheduler_gamma)
    else:
        num_training_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

    lploss = LpLoss(size_average=False)

    best_valid = float("inf")

    x_train, y_train = next(iter(train_loader))
    x = x_train.to(device)
    y = y_train.to(device)
    x_valid, y_valid = next(iter(valid_loader))
    if args.verbose:
        print(f"{args.model_type}; Input shape: {x.shape}, Target shape: {y.shape}")
    if args.strategy == "oneshot":
        assert x_train[0].shape == torch.Size([Sy, Sx, T, T_in, num_channels]), x_train[0].shape
        assert y_train[0].shape == torch.Size([Sy, Sx, T, num_channels]), y_train[0].shape
        assert x_valid[0].shape == torch.Size([Sy, Sx, T, T_in, num_channels]), x_valid[0].shape
        assert y_valid[0].shape == torch.Size([Sy, Sx, T, num_channels]), y_valid[0].shape
    elif args.strategy == "markov":
        assert x_train[0].shape == torch.Size([Sy, Sx, 1, num_channels]), x_train[0].shape
        assert y_train[0].shape == torch.Size([Sy, Sx, num_channels]), y_train[0].shape
        assert x_valid[0].shape == torch.Size([Sy, Sx, 1, num_channels]), x_valid[0].shape
        assert y_valid[0].shape == torch.Size([Sy, Sx, T, num_channels]), y_valid[0].shape
    else:  # strategy == recurrent or teacher_forcing
        assert x_train[0].shape == torch.Size([Sy, Sx, T_in, num_channels]), x_train[0].shape
        assert x_valid[0].shape == torch.Size([Sy, Sx, T_in, num_channels]), x_valid[0].shape
        assert y_valid[0].shape == torch.Size([Sy, Sx, T, num_channels]), y_valid[0].shape
        if args.strategy == "recurrent":
            assert y_train[0].shape == torch.Size([Sy, Sx, T, num_channels]), y_train[0].shape
        else:  # strategy == teacher_forcing
            assert y_train[0].shape == torch.Size([Sy, Sx, num_channels]), y_train[0].shape

    model.eval()
    if args.verbose:
        print(
            f"{args.model_type} pre-train equivariance checks: Rotations - {eq_check_rt(model, x, spatial_dims)}, Reflections - {eq_check_rf(model, x, spatial_dims)}")
    start = default_timer()
    print("Training...")
    step_ct = 0
    train_times = []
    eval_times = []
    best_epoch = 0
    for ep in range(epochs):
        model.train()
        t1 = default_timer()

        train_l2 = train_vort_l2 = train_pres_l2 = 0

        for xx, yy in tqdm(train_loader, disable=not args.verbose):
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            if data_aug:  # perform data augmentation for baseline FNO
                for b in range(len(xx)):
                    for j in range(len(spatial_dims)):
                        for l in range(j + 1, len(spatial_dims)):
                            k_rt = random.randint(0, 3)  # sample an element from C_4 (the group of 90 degree rotations)
                            if k_rt > 0:
                                if not swe:  # swe from PDEARENA are not square; cannot rotate on a batch element basis
                                    dims = [spatial_dims[j] - 1, spatial_dims[l] - 1]
                                    xx[b] = xx[b].rot90(dims=dims, k=k_rt)
                                    yy[b] = yy[b].rot90(dims=dims, k=k_rt)
                                elif b == 0:
                                    dims = [spatial_dims[j], spatial_dims[l]]
                                    xx = xx.rot90(dims=dims, k=k_rt)
                                    yy = yy.rot90(dims=dims, k=k_rt)
                        if args.reflection:
                            k_rf = random.randint(0, 1)  # sample an element from D_1 (the group of reflections)
                            if k_rf == 1:
                                xx[b] = xx[b].flip(dims=(spatial_dims[j] - 1,))
                                yy[b] = yy[b].flip(dims=(spatial_dims[j] - 1,))

            if args.strategy == "recurrent":
                for t in range(yy.shape[-2]):
                    y = yy[..., t, :]
                    im = model(xx)
                    loss += lploss(im.reshape(len(im), -1, num_channels), y.reshape(len(y), -1, num_channels))
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)
                loss /= yy.shape[-2]
            else:
                im = model(xx)
                if args.strategy == "oneshot":
                    im = im.squeeze(-1)
                loss = lploss(im.reshape(len(im), -1, num_channels), yy.reshape(len(yy), -1, num_channels))

            train_l2 += loss.item()
            if swe:
                train_vort_l2 += lploss(im[..., VORT_IND].reshape(len(im), -1, 1),
                                        yy[..., VORT_IND].reshape(len(yy), -1, 1)).item()
                train_pres_l2 += lploss(im[..., PRES_IND].reshape(len(im), -1, 1),
                                        yy[..., PRES_IND].reshape(len(yy), -1, 1)).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not args.step:
                scheduler.step()
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], step_ct)
            step_ct += 1
        if args.step:
            scheduler.step()

        train_times.append(default_timer() - t1)

        # validation
        valid_l2 = valid_vort_l2 = valid_pres_l2 = 0
        valid_loss_by_channel = None
        with torch.no_grad():
            model.eval()
            # model(xx)
            for xx, yy in valid_loader:

                xx = xx.to(device)
                yy = yy.to(device)

                if valid_l2 == 0:
                    writer.add_scalar("Valid/Rotation", eq_check_rt(model, xx, spatial_dims), ep)
                    writer.add_scalar("Valid/Reflection", eq_check_rf(model, xx, spatial_dims), ep)

                pred = get_eval_pred(model=model, x=xx, strategy=args.strategy,
                                     T=T, times=eval_times).view(len(xx), Sy, Sx, T, num_channels)

                if rdb:
                    pred[..., 0] /= pred[..., 2]
                    pred[..., 1] /= pred[..., 2]
                    yy[..., 0] /= yy[..., 2]
                    yy[..., 1] /= yy[..., 2]
                valid_l2 += lploss(pred.reshape(len(pred), -1, num_channels),
                                   yy.reshape(len(yy), -1, num_channels)).item()
                if swe:
                    valid_vort_l2 += lploss(pred[..., VORT_IND].reshape(len(pred), -1, 1),
                                            yy[..., VORT_IND].reshape(len(yy), -1, 1)).item()
                    valid_pres_l2 += lploss(pred[..., PRES_IND].reshape(len(pred), -1, 1),
                                            yy[..., PRES_IND].reshape(len(yy), -1, 1)).item()

        t2 = default_timer()
        if args.verbose:
            print(f"Ep: {ep}, time: {t2 - t1}, train: {train_l2 / ntrain}, valid: {valid_l2 / nvalid}")

        writer.add_scalar("Train/Loss", train_l2 / ntrain, ep)
        writer.add_scalar("Valid/Loss", valid_l2 / nvalid, ep)
        if swe:
            writer.add_scalar("Train Vorticity/Loss", train_vort_l2 / ntrain, ep)
            writer.add_scalar("Train Pressure/Loss", train_pres_l2 / ntrain, ep)
            writer.add_scalar("Valid Vorticity/Loss", valid_vort_l2 / nvalid, ep)
            writer.add_scalar("Valid Pressure/Loss", valid_pres_l2 / nvalid, ep)

        if valid_l2 < best_valid:
            best_epoch = ep
            best_valid = valid_l2
            torch.save(model.state_dict(), path_model)
        if args.early_stopping:
            if ep - best_epoch > args.early_stopping:
                break

    stop = default_timer()
    train_time = stop - start
    train_times = torch.tensor(train_times).mean().item()
    num_eval = len(eval_times)
    eval_times = torch.tensor(eval_times).mean().item()
    model.eval()
    if args.verbose:
        print(
            f"{args.model_type} post-train equivariance checks: Rotations - {eq_check_rt(model, xx, spatial_dims)}, Reflections - {eq_check_rf(model, xx, spatial_dims)}")

    # test
    model.load_state_dict(torch.load(path_model))
    model.eval()
    test_l2 = test_vort_l2 = test_pres_l2 = 0
    rotations_l2 = 0
    reflections_l2 = 0
    test_rt_l2 = 0
    test_rf_l2 = 0
    test_loss_by_channel = None
    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            pred = get_eval_pred(model=model, x=xx, strategy=args.strategy,
                                 T=T, times=[]).view(len(xx), Sy, Sx, T, num_channels)
            if rdb:
                pred[..., 0] /= pred[..., 2]
                pred[..., 1] /= pred[..., 2]
                yy[..., 0] /= yy[..., 2]
                yy[..., 1] /= yy[..., 2]
            test_l2 += lploss(pred.reshape(len(pred), -1, num_channels), yy.reshape(len(yy), -1, num_channels)).item()

            if swe:
                test_vort_l2 += lploss(pred[..., VORT_IND].reshape(len(pred), -1, 1),
                                       yy[..., VORT_IND].reshape(len(yy), -1, 1)).item()
                test_pres_l2 += lploss(pred[..., PRES_IND].reshape(len(pred), -1, 1),
                                       yy[..., PRES_IND].reshape(len(yy), -1, 1)).item()

            rotations_l2 += lploss(model(xx).rot90(dims=list(spatial_dims)[:2]).reshape(len(pred), -1, num_channels),
                                   model(xx.rot90(dims=list(spatial_dims)[:2])).reshape(len(pred), -1, num_channels))
            reflections_l2 += lploss(model(xx).flip(dims=(spatial_dims[0],)).reshape(len(pred), -1, num_channels),
                                     model(xx.flip(dims=(spatial_dims[0],))).reshape(len(pred), -1, num_channels))

        for xx, yy in test_rt_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            pred = get_eval_pred(model=model, x=xx, strategy=args.strategy,
                                 T=T, times=[]).view(len(xx), Sy, Sx, T, num_channels)
            if rdb:
                pred[..., 0] /= pred[..., 2]
                pred[..., 1] /= pred[..., 2]
                yy[..., 0] /= yy[..., 2]
                yy[..., 1] /= yy[..., 2]
            test_rt_l2 += lploss(pred.reshape(len(pred), -1, num_channels),
                                 yy.reshape(len(yy), -1, num_channels)).item()

        for xx, yy in test_rf_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            pred = get_eval_pred(model=model, x=xx, strategy=args.strategy,
                                 T=T, times=[]).view(len(xx), Sy, Sx, T, num_channels)
            if rdb:
                pred[..., 0] /= pred[..., 2]
                pred[..., 1] /= pred[..., 2]
                yy[..., 0] /= yy[..., 2]
                yy[..., 1] /= yy[..., 2]
            test_rf_l2 += lploss(pred.reshape(len(pred), -1, num_channels),
                                 yy.reshape(len(yy), -1, num_channels)).item()
        rotations_l2 = rotations_l2 / ntest
        reflections_l2 = reflections_l2 / ntest
        writer.add_scalar("Test/Rotation", test_rt_l2 / ntest, best_epoch)
        writer.add_scalar("Test/Reflection", test_rf_l2 / ntest, best_epoch)
        writer.add_scalar("Test/Loss", test_l2 / ntest, best_epoch)
        if swe:
            writer.add_scalar("Test Vorticity/Loss", test_vort_l2 / ntest, best_epoch)
            writer.add_scalar("Test Pressure/Loss", test_pres_l2 / ntest, best_epoch)

    test_time_l2 = test_space_l2 = ntest_super = test_int_space_l2 = test_int_time_l2 = None
    if args.super:
        if args.super_path and full_data is None:  # FNO data
            indent = 1
            try:
                with h5py.File(args.super_path, 'r') as f:
                    data = np.array(f['u'])
                data = np.transpose(data, axes=range(len(data.shape) - 1, -1, -1))
            except:
                data = scipy.io.loadmat(os.path.expandvars(args.super_path))['u'].astype(np.float32)

            if args.nsuper:
                data = data[:args.nsuper]

            assert data.shape[1] == S_super, "wrong super space"
            assert data.shape[2] == S_super, "wrong super space"

            # prepare inputs and target for space and time super res
            test_a = data[..., 3:T_in * 4:4]
            test_space_u = data[..., T_in * 4:(T + T_in) * 4:4]
            test_time_u = data[..., T_in * 4:(T + T_in) * 4]

            assert test_time_u.shape[-1] == T_super, "wrong super time"

            test_space = torch.from_numpy(np.concatenate([test_a, test_space_u], axis=-1)).unsqueeze(-1)
            test_time = torch.from_numpy(np.concatenate([test_a, test_time_u], axis=-1)).unsqueeze(-1)

        elif full_data is not None:  # otherwise, SWE data

            if args.nsuper:
                full_data = full_data[:args.nsuper]

            if rdb or ns:  # SWE
                test_space = torch.from_numpy(full_data[..., ::4, :])

            test_time = np.concatenate([full_data[..., ::4, :][..., :1, :], full_data[..., 4:, :]], axis=-2)
            test_time = torch.from_numpy(test_time)
        else:
            raise ValueError("Missing super data")

        test_int_space = test_space.clone()
        test_int_time = test_time.clone()

        batch_size = 1

        test_space = pde_data(test_space, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
        test_int_space = pde_data(test_int_space, train=False, strategy=args.strategy, T_in=T_in, T_out=T)

        test_time = pde_data(test_time, train=False, strategy=args.strategy, T_in=T_in, T_out=T_super)
        test_int_time = pde_data(test_int_time, train=False, strategy=args.strategy, T_in=T_in, T_out=T_super)

        space_loader = torch.utils.data.DataLoader(test_space, batch_size=batch_size, shuffle=False)
        space_int_loader = torch.utils.data.DataLoader(test_int_space, batch_size=batch_size, shuffle=False)

        time_loader = torch.utils.data.DataLoader(test_time, batch_size=batch_size, shuffle=False)
        time_int_loader = torch.utils.data.DataLoader(test_int_time, batch_size=batch_size, shuffle=False)

        ntest_super = len(space_loader)

        test_time_l2 = 0
        test_int_time_l2 = 0

        test_space_l2 = 0
        test_int_space_l2 = 0

        space_permute_inds = [0, 3, 1, 2]
        space_unpermute_inds = [0, 2, 3, 1]
        space_int_size = [*(S_super,) * d]

        time_permute_inds = [0, 4, 1, 2, 3]
        time_unpermute_inds = [0, 2, 3, 4, 1]
        time_int_size = [*(S_super,) * d, T_super]

        with torch.no_grad():
            for xx, yy in space_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                pred = get_eval_pred(model=model, x=xx, strategy=args.strategy,
                                     T=T, times=[]).view(len(xx), *(S_super,) * d, T, num_channels)
                if rdb:
                    pred[..., 0] /= pred[..., 2]
                    pred[..., 1] /= pred[..., 2]
                    yy[..., 0] /= yy[..., 2]
                    yy[..., 1] /= yy[..., 2]
                test_space_l2 += lploss(pred.reshape(len(pred), -1, num_channels),
                                        yy.reshape(len(yy), -1, num_channels)).item()

            for xx, yy in space_int_loader:
                if rdb:
                    xx = sampler(xx.view(1, S_super, S_super, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).view(
                        (1, *x_shape[1:])).to(device)
                else:
                    xx = xx[:, ::4, ::4].to(device)
                yy = yy.to(device)
                pred = get_eval_pred(model=model, x=xx, strategy=args.strategy,
                                     T=T, times=[]).reshape(len(xx), *(S,) * d, -1)
                pred = torch.nn.functional.interpolate(pred.permute(space_permute_inds), size=space_int_size,
                                                       mode="bilinear").permute(space_unpermute_inds)
                if rdb:
                    pred[..., 0] /= pred[..., 2]
                    pred[..., 1] /= pred[..., 2]
                    yy[..., 0] /= yy[..., 2]
                    yy[..., 1] /= yy[..., 2]
                test_int_space_l2 += lploss(pred.reshape(len(pred), -1, num_channels),
                                            yy.reshape(len(yy), -1, num_channels)).item()

            for xx, yy in time_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                pred = get_eval_pred(model=model, x=xx, strategy=args.strategy, T=T_super, times=[]
                                     ).view(len(xx), *(S_super,) * d, T_super, num_channels)
                if rdb:
                    pred[..., 0] /= pred[..., 2]
                    pred[..., 1] /= pred[..., 2]
                    yy[..., 0] /= yy[..., 2]
                    yy[..., 1] /= yy[..., 2]
                test_time_l2 += lploss(pred.reshape(len(pred), -1, num_channels),
                                       yy.reshape(len(yy), -1, num_channels)).item()

            x_new_shape = x_shape
            if threeD:
                x_new_shape[len(spatial_dims) + 1] = T_super
            x_new_shape[0] = 1
            for xx, yy in time_int_loader:
                if rdb:
                    xx = sampler(xx.view(1, S_super, S_super, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).view(
                        x_new_shape).to(device)
                else:
                    xx = xx[:, ::4, ::4].to(device)
                if threeD:
                    xx = xx[:, :, :, ::4]

                yy = yy.to(device)
                pred = get_eval_pred(model=model, x=xx, strategy=args.strategy,
                                     T=T, times=[]).view(len(xx), *(S,) * d, T, num_channels)
                pred = torch.nn.functional.interpolate(pred.permute(time_permute_inds), size=time_int_size,
                                                       mode="trilinear").permute(time_unpermute_inds)
                if rdb:
                    pred[..., 0] /= pred[..., 2]
                    pred[..., 1] /= pred[..., 2]
                    yy[..., 0] /= yy[..., 2]
                    yy[..., 1] /= yy[..., 2]
                test_int_time_l2 += lploss(pred.reshape(len(pred), -1, num_channels),
                                           yy.reshape(len(yy), -1, num_channels)).item()

        test_space_l2 = test_space_l2 / ntest_super
        writer.add_scalar("Super Space Test/Loss", test_space_l2, best_epoch)
        test_int_space_l2 = test_int_space_l2 / ntest_super
        writer.add_scalar("Super Space Interpolation Test/Loss", test_int_space_l2, best_epoch)

        test_time_l2 = test_time_l2 / ntest_super
        writer.add_scalar("Super Time Test/Loss", test_time_l2, best_epoch)
        test_int_time_l2 = test_int_time_l2 / ntest_super
        writer.add_scalar("Super Time Interpolation Test/Loss", test_int_time_l2, best_epoch)

    print(
        f"{args.model_type} done training; \nTest: {test_l2 / ntest}, Rotations: {rotations_l2}, Reflections: {reflections_l2}, Super Space Test: {test_space_l2}, Super Time Test: {test_time_l2}")
    summary = f"Args: {str(args)}" \
              f"\nParameters: {complex_ct}" \
              f"\nTrain time: {train_time}" \
              f"\nMean epoch time: {train_times}" \
              f"\nMean inference time: {eval_times}" \
              f"\nNum inferences: {num_eval}" \
              f"\nTrain: {train_l2 / ntrain}" \
              f"\nValid: {valid_l2 / nvalid}" \
              f"\nTest: {test_l2 / ntest}" \
              f"\nRotation Test: {test_rt_l2 / ntest}" \
              f"\nReflection Test: {test_rf_l2 / ntest}" \
              f"\nSuper Space Test: {test_space_l2}" \
              f"\nSuper Space Interpolation Test: {test_int_space_l2}" \
              f"\nSuper S: {S_super}" \
              f"\nSuper Time Test: {test_time_l2}" \
              f"\nSuper Time Interpolation Test: {test_int_time_l2}" \
              f"\nSuper T: {T_super}" \
              f"\nBest Valid: {best_valid / nvalid}" \
              f"\nBest epoch: {best_epoch + 1}" \
              f"\nTest Rotation Equivariance loss: {rotations_l2}" \
              f"\nTest Reflection Equivariance loss: {reflections_l2}" \
              f"\nEpochs trained: {ep}"
    if swe:
        summary += f"\nVorticity Test: {test_vort_l2 / ntest}" \
                   f"\nPressure Test: {test_pres_l2 / ntest}"
    txt = "results"
    if args.txt_suffix:
        txt += f"_{args.txt_suffix}"
    txt += ".txt"

    with open(os.path.join(root, txt), 'w') as f:
        f.write(summary)
    writer.flush()
    writer.close()

    # f = open("res_rdb3d_%s_%s_n%d.txt" % (args.model_type, grid_type, args.ntrain), "a")
    f = open("res_%s_%s_n%d.txt" % (args.model_type, grid_type, args.ntrain), "a")
    f.write(f'{args.learning_rate:e}, {args.lmbda:e}, {args.seed:d}, {train_times}, {best_epoch + 1}, '
            f'{train_l2 / ntrain}, {best_valid / nvalid}, {test_l2 / ntest}\n')
    f.close()


if __name__ == "__main__":

    ################################################################
    # configs
    ################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str, default="./results/tmp", help="path to store results")
    parser.add_argument("--suffix", type=str, default=None, help="suffix to add to the results path")
    parser.add_argument("--txt_suffix", type=str, default=None, help="suffix to add to the results txt")
    parser.add_argument("--data_path", type=str, required=True, help="path to the data")
    parser.add_argument("--super_path", type=str, default=None, help="path to the superresolution data")
    parser.add_argument("--super", action="store_true", help="enable superres testing")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--T", type=int, required=True, help="number of timesteps to predict")
    parser.add_argument("--ntrain", type=int, required=True, help="training sample size")
    parser.add_argument("--nvalid", type=int, required=True, help="valid sample size")
    parser.add_argument("--ntest", type=int, required=True, help="test sample size")
    parser.add_argument("--nsuper", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--Gwidth", type=int, default=10,
                        help="hidden dimension of equivariant layers if model_type=hybrid")
    parser.add_argument("--n_equiv", type=int, default=3, help="number of equivariant layers if model_type=hybrid")
    parser.add_argument("--reflection", action="store_true", help="symmetry group p4->p4m for data augmentation")
    parser.add_argument("--grid", type=str, default=None, help="[symmetric, cartesian, None]")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping", type=int, default=None,
                        help="stop if validation error does not improve for successive epochs")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--step", action="store_true", help="use step scheduler")
    parser.add_argument("--gamma", type=float, default=None, help="gamma for step scheduler")
    parser.add_argument("--step_size", type=int, default=None, help="step size for step scheduler")
    parser.add_argument("--lmbda", type=float, default=1e-4, help="weight decay for adam")
    parser.add_argument("--strategy", type=str, default="markov", help="markov, recurrent or oneshot")
    parser.add_argument("--time_pad", action="store_true", help="pad the time dimension for strategy=oneshot")
    parser.add_argument("--noise_std", type=float, default=0.00, help="amount of noise to inject for strategy=markov")
    parser.add_argument("--seed_set", type=int, default=0)

    args = parser.parse_args()
    print(args)

    if args.model_type in ("FNO2d", "FNO2d_aug", "FNO3d", "FNO3d_aug", "GFNO2d_p4", "GFNO2d_p4m", "GFNO3d_p4",
                           "GFNO3d_p4m", "radialNO2d_p4", "radialNO2d_p4m", "radialNO3d_p4", "radialNO3d_p4m",
                           "Unet_Rot2d", "Unet_Rot_M2d", "Unet_Rot_3D", "UNet2d", "UNet3d"):
        lrs = [1e-3]
        wds = [1e-4]
    else:
        # best lr and wd
        # ns2d:
        if args.ntrain == 1000:
            if args.model_type == "clawFNO2d":
                lrs = [1e-2]
                wds = [1e-4]
            elif args.model_type == "clawGFNO2d_p4":
                lrs = [3e-3]
                wds = [3e-4]
        elif args.ntrain == 100:
            if args.model_type == "clawFNO2d":
                lrs = [3e-2]
                wds = [3e-4]
            elif args.model_type == "clawGFNO2d_p4":
                lrs = [1e-2]
                wds = [3e-4]
        elif args.ntrain == 10:
            if args.model_type == "clawFNO2d":
                lrs = [3e-2]
                wds = [3e-4]
            elif args.model_type == "clawGFNO2d_p4":
                lrs = [1e-2]
                wds = [3e-4]
        # rdb3d:
        # if args.ntrain == 100:
        #     if args.model_type == "clawFNO3d":
        #         lrs = [1e-2]
        #         wds = [1e-6]
        #     elif args.model_type == "clawGFNO3d_p4":
        #         lrs = [1e-2]
        #         wds = [1e-6]
        # elif args.ntrain == 10:
        #     if args.model_type == "clawFNO3d":
        #         lrs = [3e-2]
        #         wds = [3e-7]
        #     elif args.model_type == "clawGFNO3d_p4":
        #         lrs = [1e-2]
        #         wds = [1e-6]
        # elif args.ntrain == 2:
        #     if args.model_type == "clawFNO3d":
        #         lrs = [3e-2]
        #         wds = [1e-5]
        #     elif args.model_type == "clawGFNO3d_p4":
        #         lrs = [1e-2]
        #         wds = [3e-5]

    seeds = [0]

    grid_type = "symmetric"
    if args.grid:
        grid_type = args.grid
        assert grid_type in ['symmetric', 'cartesian', 'None']

    if args.ntrain == 10:
        args.batch_size = 2
    if args.ntrain < 10:
        args.batch_size = 1

    if args.seed_set == 0:
        f = open("res_%s_%s_n%d.txt" % (args.model_type, grid_type, args.ntrain), "w")
        f.write(f'learning_rate, wd, seed, mean epoch time, best epoch, train_loss, valid_loss, test_loss\n')
        f.close()

    icount = 0
    icount_total = len(lrs) * len(wds) * len(seeds)
    for iseed in seeds:
        for lr in lrs:
            for wd in wds:
                icount += 1
                args.learning_rate = lr
                args.lmbda = wd
                args.seed = iseed

                print("-" * 100)
                print(f'>> Running {icount}/{icount_total}, lr: {lr:e}, wd: {wd:e}, seed: {iseed}')
                print("-" * 100)
                main(args)

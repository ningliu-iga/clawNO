import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utils_mat_cir import *
from egcl_mat_cir import E_GCL_GKN
from timeit import default_timer
import os
import sys
import bisect

iseed = 0
torch.manual_seed(iseed)
np.random.seed(iseed)
torch.cuda.manual_seed(iseed)
# torch.backends.cudnn.deterministic = True


class EGKN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1, device='cuda:0', act_fn=torch.nn.ReLU()):
        super().__init__()
        self.depth = depth
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)
        kernel = DenseNet([ker_in, ker_width // 2, ker_width, width ** 2], torch.nn.ReLU)
        self.egkn_conv = E_GCL_GKN(width, width, width, kernel, depth, act_fn=act_fn)
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(width, width * 2), act_fn, torch.nn.Linear(width * 2, out_width))
        smdx, smdy = OBM.build_matrix()
        self.smdx = torch.tensor(smdx, device=device, dtype=torch.float32)
        self.smdy = torch.tensor(smdy, device=device, dtype=torch.float32)
        self.OBMmatrix = {'smdx':self.smdx,'smdy':self.smdy}
        self.ID = torch.tensor(OBM.ID, device=device, dtype=torch.float32)
        self.batch_size = OBM.batch_size


    def forward(self, data):
        h, edge_index, edge_attr, coords_curr = data.x, data.edge_index, data.edge_attr, \
                                                data.coords_init.detach().clone()
        h = self.fc1(h)
        for k in range(self.depth):
            h, coords_curr = self.egkn_conv(h, edge_index, coords_curr, edge_attr, self.OBMmatrix)
        h = self.fc2(h)
        return h, coords_curr


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


if __name__ == '__main__':
    ################################################################
    # set hyperparameters
    ################################################################
    ntrain = int(sys.argv[1])
    # 3 * 5 * 10 = 30 * 5
    if ntrain == 2:
        lrs = [1e-2]
        gammas = [0.9]
        wds = [1e-3]
    elif ntrain == 20:
        lrs = [1e-2]
        gammas = [0.9]
        wds = [3e-7]
    elif ntrain == 100:
        lrs = [1e-2]
        gammas = [0.5]
        wds = [3e-3]
    
    seeds = [0]

    f = open("res_clawINO_lps_n%d_seeds.txt" % ntrain, "w")
    f.write(f'seed, ntrain, lr, gamma, w_d, train_loss, valid_loss, test_loss, best_epochs\n')
    f.close()

    # specify data path
    #TRAIN_PATH = './data_linear_peridynamic_solid/domain1_0.05_lognorm_a2_t3.mat'
    # TRAIN_PATH = './data/domain1_0.05_lognorm_a2_t3.mat'
    TRAIN_PATH = './data/incomp_2d_dri_order2.mat'
    TEST_PATH = TRAIN_PATH
    nnodes_bd = 53
    nnodes_total = 246

    nvalid = 100
    ntest = 100

    is_minus_aver_bc = False

    ################################################################
    # load data
    ################################################################

    reader = MatReader(TRAIN_PATH)
    all_u = reader.read_field('u')
    coords = reader.read_field('coords')
    #b = reader.read_field('b')
    all_f = reader.read_field('f')

    # specify ball radius
    radius_train = 1
    radius_test = radius_train

    # model and training parameters
    batch_size = 1 if ntrain <= 5 else 2
    batch_size2 = batch_size
    width = 64
    ker_width = 1024
    edge_features = 4  # 1-2 features for |x-y|_1, |x-y|_2, 3-4 features for |f(x)| and |f(y)|
    node_features = 2  # number of kinds of data used for training
    out_features = 1

    layer_end = 3

    epochs = 500
    scheduler_step = 50

    # setup OBMeshfree stuff
    radius_obm=0.05*16.0
    ID = np.concatenate((np.ones((nnodes_bd)), np.zeros((nnodes_total - nnodes_bd))))
    OBM = OBMeshfree(radius_obm, coords, ID, batch_size)

    t1 = default_timer()

    # train_b = b[:ntrain, :]
    # valid_b = b[ntrain:ntrain + nvalid, :]
    # test_b = b[-ntest:, :]

    # data for training
    train_ux = all_u[:ntrain, :nnodes_total]
    train_uy = all_u[:ntrain, nnodes_total:]
    train_fx = all_f[:ntrain, :nnodes_total]
    train_fy = all_f[:ntrain, nnodes_total:]
    train_ux_bd = torch.zeros((ntrain, nnodes_total))
    train_uy_bd = torch.zeros((ntrain, nnodes_total))
    for i in range(ntrain):
        train_ux_bd[i, :nnodes_bd] = train_ux[i, :nnodes_bd].detach().clone()
        train_uy_bd[i, :nnodes_bd] = train_uy[i, :nnodes_bd].detach().clone()

    train_ux_bd_mean = torch.zeros(ntrain, 1)
    train_uy_bd_mean = torch.zeros(ntrain, 1)

    if is_minus_aver_bc:
        for i in range(ntrain):
            train_ux_bd_mean[i, 0] = train_ux_bd[i, :].nanmean()
            train_ux_bd[i, :] = torch.sub(train_ux_bd[i, :], train_ux_bd_mean[i])
            train_ux[i, :] = torch.sub(train_ux[i, :], train_ux_bd_mean[i])

            train_uy_bd_mean[i, 0] = train_uy_bd[i, :].nanmean()
            train_uy_bd[i, :] = torch.sub(train_uy_bd[i, :], train_uy_bd_mean[i])
            train_uy[i, :] = torch.sub(train_uy[i, :], train_uy_bd_mean[i])

    train_u = torch.cat((train_ux, train_uy), dim=1)
    train_u_bd = torch.cat((train_ux_bd, train_uy_bd), dim=1)
    train_u_bd_mean = torch.cat((train_ux_bd_mean, train_uy_bd_mean), dim=1)

    # norms
    train_u_bd_norm = torch.zeros(ntrain, nnodes_total)
    train_u_norm = torch.zeros(ntrain, nnodes_total)
    train_f_norm = torch.zeros(ntrain, nnodes_total)
    for i in range(ntrain):
        train_u_norm[i, :] = torch.sqrt((train_ux[i, :]) ** 2 + (train_uy[i, :]) ** 2)
        train_u_bd_norm[i, :] = torch.sqrt((train_ux_bd[i, :]) ** 2 + (train_uy_bd[i, :]) ** 2)
        train_f_norm[i, :] = torch.sqrt((train_fx[i, :]) ** 2 + (train_fy[i, :]) ** 2)

    # data for validation
    valid_ux = all_u[-ntest - nvalid:-ntest, :nnodes_total]
    valid_uy = all_u[-ntest - nvalid:-ntest, nnodes_total:]
    valid_fx = all_f[-ntest - nvalid:-ntest, :nnodes_total]
    valid_fy = all_f[-ntest - nvalid:-ntest, nnodes_total:]
    valid_ux_bd = torch.zeros((nvalid, nnodes_total))
    valid_uy_bd = torch.zeros((nvalid, nnodes_total))
    for i in range(nvalid):
        valid_ux_bd[i, :nnodes_bd] = valid_ux[i, :nnodes_bd].detach().clone()
        valid_uy_bd[i, :nnodes_bd] = valid_uy[i, :nnodes_bd].detach().clone()

    valid_ux_bd_mean = torch.zeros(nvalid, 1)
    valid_uy_bd_mean = torch.zeros(nvalid, 1)

    if is_minus_aver_bc:
        for i in range(nvalid):
            valid_ux_bd_mean[i, 0] = valid_ux_bd[i, :].nanmean()
            valid_ux_bd[i, :] = torch.sub(valid_ux_bd[i, :], valid_ux_bd_mean[i])
            valid_ux[i, :] = torch.sub(valid_ux[i, :], valid_ux_bd_mean[i])

            valid_uy_bd_mean[i, 0] = valid_uy_bd[i, :].nanmean()
            valid_uy_bd[i, :] = torch.sub(valid_uy_bd[i, :], valid_uy_bd_mean[i])
            valid_uy[i, :] = torch.sub(valid_uy[i, :], valid_uy_bd_mean[i])

    valid_u = torch.cat((valid_ux, valid_uy), dim=1)
    valid_u_bd = torch.cat((valid_ux_bd, valid_uy_bd), dim=1)
    valid_u_bd_mean = torch.cat((valid_ux_bd_mean, valid_uy_bd_mean), dim=1)

    # norms
    valid_u_bd_norm = torch.zeros(nvalid, nnodes_total)
    valid_u_norm = torch.zeros(nvalid, nnodes_total)
    valid_f_norm = torch.zeros(nvalid, nnodes_total)
    for i in range(nvalid):
        valid_u_norm[i, :] = torch.sqrt((valid_ux[i, :]) ** 2 + (valid_uy[i, :]) ** 2)
        valid_u_bd_norm[i, :] = torch.sqrt((valid_ux_bd[i, :]) ** 2 + (valid_uy_bd[i, :]) ** 2)
        valid_f_norm[i, :] = torch.sqrt((valid_fx[i, :]) ** 2 + (valid_fy[i, :]) ** 2)

    # data for test
    test_ux = all_u[-ntest:, :nnodes_total]
    test_uy = all_u[-ntest:, nnodes_total:]
    test_fx = all_f[-ntest:, :nnodes_total]
    test_fy = all_f[-ntest:, nnodes_total:]
    test_ux_bd = torch.zeros((ntest, nnodes_total))
    test_uy_bd = torch.zeros((ntest, nnodes_total))
    for i in range(ntest):
        test_ux_bd[i, :nnodes_bd] = test_ux[i, :nnodes_bd].detach().clone()
        test_uy_bd[i, :nnodes_bd] = test_uy[i, :nnodes_bd].detach().clone()

    test_ux_bd_mean = torch.zeros(ntest, 1)
    test_uy_bd_mean = torch.zeros(ntest, 1)

    if is_minus_aver_bc:
        for i in range(ntest):
            test_ux_bd_mean[i, 0] = test_ux_bd[i, :].nanmean()
            test_ux_bd[i, :] = torch.sub(test_ux_bd[i, :], test_ux_bd_mean[i])
            test_ux[i, :] = torch.sub(test_ux[i, :], test_ux_bd_mean[i])

            test_uy_bd_mean[i, 0] = test_uy_bd[i, :].nanmean()
            test_uy_bd[i, :] = torch.sub(test_uy_bd[i, :], test_uy_bd_mean[i])
            test_uy[i, :] = torch.sub(test_uy[i, :], test_uy_bd_mean[i])

    test_u = torch.cat((test_ux, test_uy), dim=1)
    test_u_bd = torch.cat((test_ux_bd, test_uy_bd), dim=1)
    test_u_bd_mean = torch.cat((test_ux_bd_mean, test_uy_bd_mean), dim=1)

    # norms
    test_u_bd_norm = torch.zeros(ntest, nnodes_total)
    test_u_norm = torch.zeros(ntest, nnodes_total)
    test_f_norm = torch.zeros(ntest, nnodes_total)
    for i in range(ntest):
        test_u_norm[i, :] = torch.sqrt((test_ux[i, :]) ** 2 + (test_uy[i, :]) ** 2)
        test_u_bd_norm[i, :] = torch.sqrt((test_ux_bd[i, :]) ** 2 + (test_uy_bd[i, :]) ** 2)
        test_f_norm[i, :] = torch.sqrt((test_fx[i, :]) ** 2 + (test_fy[i, :]) ** 2)

    # b_normalizer = GaussianNormalizer(train_b)
    bd_normalizer = GaussianNormalizer(train_u_bd_norm)
    f_normalizer = GaussianNormalizer(train_f_norm)

    # train_b = b_normalizer.encode(train_b)
    # valid_b = b_normalizer.encode(valid_b)
    # test_b = b_normalizer.encode(test_b)
    train_u_bd_norm = bd_normalizer.encode(train_u_bd_norm)
    valid_u_bd_norm = bd_normalizer.encode(valid_u_bd_norm)
    test_u_bd_norm = bd_normalizer.encode(test_u_bd_norm)
    train_f_norm = f_normalizer.encode(train_f_norm)
    valid_f_norm = f_normalizer.encode(valid_f_norm)
    test_f_norm = f_normalizer.encode(test_f_norm)

    u_normalizer = GaussianNormalizer(train_u_norm)
    train_u_norm = u_normalizer.encode(train_u_norm)

    meshgenerator = IrregularMeshGenerator(coords)
    edge_index = meshgenerator.ball_connectivity(radius_train)

    # generate initial grid (grid+u_bd) and final grid (grid+u)
    grid_init = torch.zeros(ntrain, nnodes_total, 2)
    grid_final = torch.zeros(ntrain, nnodes_total, 2)
    for i in range(ntrain):
        grid_init[i, :, 0] = coords[:, 0] + train_ux_bd[i, :]
        grid_init[i, :, 1] = coords[:, 1] + train_uy_bd[i, :]
        grid_final[i, :, 0] = coords[:, 0] + train_ux[i, :]
        grid_final[i, :, 1] = coords[:, 1] + train_uy[i, :]
    data_train = []
    for j in range(ntrain):
        edge_attr = meshgenerator.attributes(theta=train_f_norm[j, :])
        data_train.append(Data(x=torch.cat([train_u_bd_norm[j, :].reshape(-1, 1),
                                            train_f_norm[j,:].reshape(-1,1)
                                            ], dim=1),
                               y=train_u_norm[j, :],
                               y_mean=train_u_bd_mean[j, :],
                               edge_index=edge_index,
                               edge_attr=edge_attr,
                               coords=coords,
                               coords_init=grid_init[j, :, :],
                               coords_final=grid_final[j, :, :],
                               ubd=torch.cat([train_ux_bd[j, :].unsqueeze(1), train_uy_bd[j,:].unsqueeze(1)] ,dim=1)  #add bd data for subtraction in loss
                               ))

    # validation
    grid_init = torch.zeros(nvalid, nnodes_total, 2)
    grid_final = torch.zeros(nvalid, nnodes_total, 2)
    for i in range(nvalid):
        grid_init[i, :, 0] = coords[:, 0] + valid_ux_bd[i, :]
        grid_init[i, :, 1] = coords[:, 1] + valid_uy_bd[i, :]
        grid_final[i, :, 0] = coords[:, 0] + valid_ux[i, :]
        grid_final[i, :, 1] = coords[:, 1] + valid_uy[i, :]
    data_valid = []
    for j in range(nvalid):
        edge_attr = meshgenerator.attributes(theta=valid_f_norm[j, :])
        data_valid.append(Data(x=torch.cat([valid_u_bd_norm[j, :].reshape(-1, 1),
                                            valid_f_norm[j,:].reshape(-1,1)
                                            ], dim=1),
                               y=valid_u_norm[j, :],
                               y_mean=valid_u_bd_mean[j, :],
                               edge_index=edge_index,
                               edge_attr=edge_attr,
                               coords=coords,
                               coords_init=grid_init[j, :, :],
                               coords_final=grid_final[j, :, :],
                               ubd=torch.cat([valid_ux_bd[j, :].unsqueeze(1), valid_uy_bd[j,:].unsqueeze(1)], dim=1)
                               ))

    # test
    grid_init = torch.zeros(ntest, nnodes_total, 2)
    grid_final = torch.zeros(ntest, nnodes_total, 2)
    for i in range(ntest):
        grid_init[i, :, 0] = coords[:, 0] + test_ux_bd[i, :]
        grid_init[i, :, 1] = coords[:, 1] + test_uy_bd[i, :]
        grid_final[i, :, 0] = coords[:, 0] + test_ux[i, :]
        grid_final[i, :, 1] = coords[:, 1] + test_uy[i, :]
    data_test = []
    for j in range(ntest):
        edge_attr = meshgenerator.attributes(theta=test_f_norm[j, :])
        data_test.append(Data(x=torch.cat([test_u_bd_norm[j, :].reshape(-1, 1),
                                            test_f_norm[j,:].reshape(-1,1)
                                            ], dim=1),
                              y=test_u_norm[j, :],
                              y_mean=test_u_bd_mean[j, :],
                              edge_index=edge_index,
                              edge_attr=edge_attr,
                              coords=coords,
                              coords_init=grid_init[j, :, :],
                              coords_final=grid_final[j, :, :],
                              ubd=torch.cat([test_ux_bd[j, :].unsqueeze(1), test_uy_bd[j,:].unsqueeze(1)], dim=1)
                              ))

    print(f'>> grid: {coords.shape}, edge_index: {edge_index.shape}, edge_attr: {edge_attr.shape}')

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=batch_size2, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

    ##################################################################################################
    #                                        training
    ##################################################################################################
    t2 = default_timer()

    print(f'>> Preprocessing completed, time elapsed: {(t2 - t1): .2f}s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

    if torch.cuda.is_available():
        u_normalizer.cuda()
    else:
        u_normalizer.cpu()

    for iseed in seeds:
        for learning_rate in lrs:
            for scheduler_gamma in gammas:
                for weight_decay in wds:

                    torch.manual_seed(iseed)
                    np.random.seed(iseed)
                    torch.cuda.manual_seed(iseed)

                    print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, w_d: {weight_decay}')

                    op_type = 'lps_cir'
                    base_dir = './clawino_%s_seed%d/n%d_lr%e_dr%f_wd%e' % (
                    op_type, iseed, ntrain, learning_rate, scheduler_gamma, weight_decay)
                    if not os.path.exists(base_dir):
                        os.makedirs(base_dir)

                    myloss = LpLoss(size_average=False)
                    best_epoch = np.zeros((layer_end,))
                    bl_train, bl_train_c, bl_valid, bl_valid_c, bl_test, bl_test_c = [], [], [], [], [], []

                    for i in range(layer_end):
                        print("-" * 100)
                        layer = 2 ** i
                        model = EGKN(width, ker_width, layer, edge_features, node_features, out_features).to(device)
                        print(f'>> Total number of model parameters: {count_params(model)}')
                        if layer != 1:
                            restart_layer = layer // 2
                            model_filename_res = '%s/model_depth%d.ckpt' % (base_dir, restart_layer)
                            if torch.cuda.is_available():
                                model.load_state_dict(torch.load(model_filename_res))
                            else:
                                model.load_state_dict(torch.load(model_filename_res, map_location='cpu'))

                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        model_filename = '%s/model_depth%d.ckpt' % (base_dir, layer)
                        ttrain, ttrain_disp, tvalid, tvalid_disp, ttest = [], [], [], [], []
                        best_train_loss = best_train_coords_loss = best_valid_loss = best_valid_coords_loss = \
                            best_test_loss = best_test_coords_loss = 1e8
                        early_stop = 0
                        for ep in range(epochs):
                            model.train()
                            optimizer = scheduler(optimizer,
                                                  LR_schedule(learning_rate, ep, scheduler_step, scheduler_gamma))
                            t1 = default_timer()
                            train_l2 = 0.0
                            train_l2_disp = 0.0
                            train_loss = 0.0
                            for batch in train_loader:
                                batch = batch.to(device)

                                optimizer.zero_grad()
                                out, dis = model(batch)
                                out = u_normalizer.decode(out.reshape(batch_size, -1))
                                y = u_normalizer.decode(batch.y.view(batch_size, -1))
                                for ii in range(batch_size):
                                    out[ii, :] = torch.add(out[ii, :], batch.y_mean[ii])
                                    y[ii, :] = torch.add(y[ii, :], batch.y_mean[ii])
                                loss = torch.norm(dis.view(-1) - batch.ubd.view(-1) - batch.coords_final.view(-1), 1)
                                loss.backward()

                                l2 = myloss(dis.view(batch_size, -1) - batch.ubd.view(batch_size, -1),
                                            batch.coords_final.view(batch_size, -1))  # add ubd
                                l2_disp = myloss(
                                    dis.view(batch_size, -1) - batch.ubd.view(batch_size, -1) - batch.coords.view(
                                        batch_size, -1),  # add ubd
                                    batch.coords_final.view(batch_size, -1) - batch.coords.view(batch_size,
                                                                                                -1))  # no need to add ubd

                                optimizer.step()
                                train_loss += loss.item()
                                train_l2 += l2.item()
                                train_l2_disp += l2_disp.item()

                            train_l2 /= ntrain
                            train_l2_disp /= ntrain
                            ttrain.append([ep, train_l2])
                            ttrain_disp.append([ep, train_l2_disp])

                            if train_l2 < best_train_loss:
                                model.eval()
                                valid_l2 = 0.0
                                valid_l2_disp = 0.0
                                with torch.no_grad():
                                    for batch in valid_loader:
                                        batch = batch.to(device)
                                        out, dis = model(batch)
                                        out = u_normalizer.decode(out.reshape(batch_size, -1))
                                        y = batch.y.view(batch_size2, -1)
                                        for ii in range(batch_size2):
                                            out[ii, :] = torch.add(out[ii, :], batch.y_mean[ii])
                                            y[ii, :] = torch.add(y[ii, :], batch.y_mean[ii])
                                        valid_l2 += myloss(dis.view(batch_size2, -1) - batch.ubd.view(batch_size2, -1),
                                                           batch.coords_final.view(batch_size2, -1)).item()  # add ubd
                                        # valid_l2 += myloss(out.view(batch_size2, -1), y.view(batch_size2, -1)).item()
                                        valid_l2_disp += myloss(dis.view(batch_size2, -1) - batch.ubd.view(batch_size2,
                                                                                                           -1) - batch.coords.view(
                                            batch_size2, -1),  # add ubd
                                                                batch.coords_final.view(batch_size2,
                                                                                        -1) - batch.coords.view(
                                                                    batch_size2, -1)).item()  # no need to add ubd

                                valid_l2 /= nvalid
                                valid_l2_disp /= nvalid
                                tvalid.append([ep, valid_l2])
                                tvalid_disp.append([ep, valid_l2_disp])

                                if valid_l2 < best_valid_loss:
                                    test_l2_disp = 0.0
                                    test_l2 = 0.0
                                    with torch.no_grad():
                                        for batch in test_loader:
                                            batch = batch.to(device)
                                            out, dis = model(batch)
                                            out = u_normalizer.decode(out.reshape(batch_size, -1))
                                            # y = batch.y.view(batch_size2, -1)
                                            # for ii in range(batch_size2):
                                            #     out[ii, :] = torch.add(out[ii, :], batch.y_mean[ii])
                                            #     y[ii, :] = torch.add(y[ii, :], batch.y_mean[ii])
                                            test_l2 += myloss(
                                                dis.view(batch_size2, -1) - batch.ubd.view(batch_size2, -1),
                                                batch.coords_final.view(batch_size2, -1)).item()  # add ubd
                                            # test_l2 += myloss(out.view(batch_size2, -1), y.view(batch_size2, -1)).item()
                                            test_l2_disp += myloss(
                                                dis.view(batch_size2, -1) - batch.ubd.view(batch_size2,
                                                                                           -1) - batch.coords.view(
                                                    batch_size2, -1),  # add ubd
                                                batch.coords_final.view(batch_size2, -1) - batch.coords.view(
                                                    batch_size2, -1)).item()  # no need to add ubd

                                    test_l2 /= ntest
                                    test_l2_disp /= ntest
                                    ttest.append([ep, test_l2])

                                    early_stop = 0
                                    best_train_loss = train_l2
                                    best_train_coords_loss = train_l2_disp
                                    best_valid_loss = valid_l2
                                    best_valid_coords_loss = valid_l2_disp
                                    best_test_loss = test_l2
                                    best_test_coords_loss = test_l2_disp
                                    best_epoch[i] = ep
                                    torch.save(model.state_dict(), model_filename)
                                    t2 = default_timer()
                                    print(
                                        f'>> depth: {layer}, epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                                        f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.4f} (disp: '
                                        f'{train_l2_disp:.4f}), valid err: {valid_l2:.4f} (disp: '
                                        f'{valid_l2_disp:.4f}), test err: {test_l2:.4f}')
                                else:
                                    early_stop += 1
                                    t2 = default_timer()
                                    print(
                                        f'>> depth: {layer}, epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                                        f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.4f} '
                                        f'(best: {best_train_loss:.4f}/{best_valid_loss:.4f}/{best_test_loss:.4f})')
                            else:
                                early_stop += 1
                                t2 = default_timer()
                                print(
                                    f'>> depth: {layer}, epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: '
                                    f'{(t2 - t1):.2f}s, train err: {train_l2:.4f} '
                                    f'(best: {best_train_loss:.4f}/{best_valid_loss:.4f}/{best_test_loss:.4f})')

                            if early_stop > 222: break
                        bl_train.append(best_train_loss)
                        bl_train_c.append(best_train_coords_loss)
                        bl_valid.append(best_valid_loss)
                        bl_valid_c.append(best_valid_coords_loss)
                        bl_test.append(best_test_loss)
                        bl_test_c.append(best_test_coords_loss)
                        with open('%s/loss_l%d_train.txt' % (base_dir, layer), 'w') as file:
                            np.savetxt(file, ttrain)
                        with open('%s/loss_l%d_train_coords.txt' % (base_dir, layer), 'w') as file:
                            np.savetxt(file, ttrain_disp)
                        with open('%s/loss_l%d_valid.txt' % (base_dir, layer), 'w') as file:
                            np.savetxt(file, tvalid)
                        with open('%s/loss_l%d_valid_coords.txt' % (base_dir, layer), 'w') as file:
                            np.savetxt(file, tvalid_disp)
                        with open('%s/loss_l%d_test.txt' % (base_dir, layer), 'w') as file:
                            np.savetxt(file, ttest)

                    print("-" * 100)
                    print("-" * 100)
                    print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, w_d: {weight_decay}')
                    print(f'>> Best train error: {best_train_loss:.4f}')
                    print(f'>> Best valid error: {best_valid_loss:.4f}')
                    print(f'>> Best test error: {best_test_loss:.4f}')
                    print(f'>> Best epoch: {best_epoch}')
                    print("-" * 100)
                    print("-" * 100)

                    f = open("res_clawINO_lps_n%d_seeds.txt" % ntrain, "a")
                    f.write(f'{iseed}, {ntrain}, {learning_rate}, {scheduler_gamma}, {weight_decay}, ')
                    f.write(','.join(str(err) for err in bl_train))
                    f.write(',')
                    f.write(','.join(str(err) for err in bl_valid))
                    f.write(',')
                    f.write(','.join(str(err) for err in bl_test))
                    f.write(f', {best_epoch}\n')
                    f.close()

    print('***************** Training Completed *****************')

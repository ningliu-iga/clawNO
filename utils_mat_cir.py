import torch
import numpy as np
import scipy.io
# import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import math
from scipy.interpolate import LinearNDInterpolator

import operator
from functools import reduce, partial

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MatReader:
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super().__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = scipy.io.loadmat(self.file_path)
            # self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):

        self.mean = torch.mean(x, 0).view(-1)
        self.std = torch.std(x, 0).view(-1)

        self.eps = eps

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.view(s)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            std = self.std[sample_idx] + self.eps  # batch * n
            mean = self.mean[sample_idx]

        s = x.size()
        x = x.view(s[0], -1)
        x = (x * std) + mean
        x = x.view(s)
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class GaussianNormalizer:
    def __init__(self, x, eps=1e-5):
        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


class LpLoss:
    def __init__(self, d=2, p=2, size_average=True, reduction=True):

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

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


class SquareMeshGenerator:
    def __init__(self, real_space, mesh_size):

        self.d = len(real_space)
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]
            # xx.ravel() is equiv to xx.reshape(-1)
            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    # the 1st col of edge_attr is the edge length for equivariance
    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                # edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
                #edge_attr = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                #                            *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T), dtype=float)
                edge_attr = None
            else:
                edge_attr = np.zeros((self.n_edges, 2))
                dist = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                                       *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T),
                                   dtype=float)

                # ang: angle based on local coordinate axis of
                # (x,y)=(self.grid[1,0]-self.grid[0,0],self.grid[1,1]-self.grid[0,1])
                ang = np.fromiter(map(lambda x1, y1, x2, y2: math.atan2(y2 - y1, x2 - x1) -
                                                             math.atan2(self.grid[1, 1] - self.grid[0, 1],
                                                                        self.grid[1, 0] - self.grid[0, 0]),
                                      *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T),
                                  dtype=float)
                # enforce angle in the range [0, 2*pi]
                for ii in range(self.n_edges):
                    if ang[ii] < 0: ang[ii] += 2 * math.pi
                    edge_attr[ii, 0] = dist[ii] * np.cos(ang[ii])
                    edge_attr[ii, 1] = dist[ii] * np.sin(ang[ii])
                # edge_attr[:, 2] = theta[self.edge_index[0]]
                # edge_attr[:, 3] = theta[self.edge_index[1]]
        else:
            # xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            #xy = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
            #                                *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T), dtype=float)
            xy = None
            if theta is None:
                # edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
                edge_attr = f(xy)
            else:
                # edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])
                edge_attr = f(xy, theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return torch.tensor(self.edge_index_boundary, dtype=torch.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3 * self.d))
                edge_attr_boundary[:, 0:2 * self.d] = self.grid[self.edge_index_boundary.T].reshape(
                    (self.n_edges_boundary, -1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d + 1] = theta[self.edge_index_boundary[1]]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            if theta is None:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index_boundary[0]],
                                       theta[self.edge_index_boundary[1]])

        return torch.tensor(edge_attr_boundary, dtype=torch.float)


class IrregularMeshGenerator:
    def __init__(self, grid):
        self.n, self.d = grid.shape
        # self.s = mesh_size[0]
        # assert len(mesh_size) == self.d
        self.grid = grid

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    # the 1st col of edge_attr is the edge length for equivariance
    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                # edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
                #edge_attr = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                #                            *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T), dtype=float)
                edge_attr = None
            else:
                edge_attr = np.zeros((self.n_edges, 4))  # equals to edge features in the main code
                dist = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                                       *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T),
                                   dtype=float)

                # ang: angle based on local coordinate axis of
                # (x,y)=(self.grid[1,0]-self.grid[0,0],self.grid[1,1]-self.grid[0,1])
                ang = np.fromiter(map(lambda x1, y1, x2, y2: math.atan2(y2 - y1, x2 - x1) -
                                                             math.atan2(self.grid[1, 1] - self.grid[0, 1],
                                                                        self.grid[1, 0] - self.grid[0, 0]),
                                      *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T), dtype=float)
                # enforce angle in the range [0, 2*pi]
                for ii in range(self.n_edges):
                    if ang[ii] < 0: ang[ii] += 2 * math.pi
                    edge_attr[ii, 0] = dist[ii] * np.cos(ang[ii])
                    edge_attr[ii, 1] = dist[ii] * np.sin(ang[ii])
                # theta here corresponds to |f| in the main code
                edge_attr[:, 2] = theta[self.edge_index[0]]
                edge_attr[:, 3] = theta[self.edge_index[1]]
        else:
            # xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            #xy = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
            #                                *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T), dtype=float)
            xy = None
            if theta is None:
                # edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
                edge_attr = f(xy)
            else:
                # edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])
                edge_attr = f(xy, theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return torch.tensor(self.edge_index_boundary, dtype=torch.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3 * self.d))
                edge_attr_boundary[:, 0:2 * self.d] = self.grid[self.edge_index_boundary.T].reshape(
                    (self.n_edges_boundary, -1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d + 1] = theta[self.edge_index_boundary[1]]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            if theta is None:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index_boundary[0]],
                                       theta[self.edge_index_boundary[1]])

        return torch.tensor(edge_attr_boundary, dtype=torch.float)


class RandomMeshGenerator(object):
    def __init__(self, real_space, mesh_size, sample_size):
        super(RandomMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_size

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        if self.m > self.n:
            self.m = self.n

        self.idx = np.array(range(self.n))
        self.grid_sample = self.grid

    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[:self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                theta = theta[self.idx]
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                theta = theta[self.idx]
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

    # def get_boundary(self):
    #     s = self.s
    #     n = self.n
    #     boundary1 = np.array(range(0, s))
    #     boundary2 = np.array(range(n - s, n))
    #     boundary3 = np.array(range(s, n, s))
    #     boundary4 = np.array(range(2 * s - 1, n, s))
    #     self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])
    #
    # def boundary_connectivity2d(self, stride=1):
    #
    #     boundary = self.boundary[::stride]
    #     boundary_size = len(boundary)
    #     vertice1 = np.array(range(self.n))
    #     vertice1 = np.repeat(vertice1, boundary_size)
    #     vertice2 = np.tile(boundary, self.n)
    #     self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
    #     self.n_edges_boundary = self.edge_index_boundary.shape[1]
    #     return torch.tensor(self.edge_index_boundary, dtype=torch.long)
    #
    # def attributes_boundary(self, f=None, theta=None):
    #     # if self.edge_index_boundary == None:
    #     #     self.boundary_connectivity2d()
    #     if f is None:
    #         if theta is None:
    #             edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #         else:
    #             edge_attr_boundary = np.zeros((self.n_edges_boundary, 3*self.d))
    #             edge_attr_boundary[:,0:2*self.d] = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #             edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
    #             edge_attr_boundary[:, 2 * self.d +1] = theta[self.edge_index_boundary[1]]
    #     else:
    #         xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #         if theta is None:
    #             edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:])
    #         else:
    #             edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index_boundary[0]], theta[self.edge_index_boundary[1]])
    #
    #     return torch.tensor(edge_attr_boundary, dtype=torch.float)


class RandomGridSplitter(object):
    def __init__(self, grid, resolution, m=200, l=2, radius=0.25):
        super(RandomGridSplitter, self).__init__()

        self.grid = grid
        self.resolution = resolution
        self.n = resolution ** 2
        self.m = m
        self.l = l
        self.radius = radius

        assert self.n % self.m == 0
        self.num = self.n // self.m

    def get_data(self, theta):

        data = []
        for i in range(self.l):
            perm = torch.randperm(self.n)
            perm = perm.reshape(self.num, self.m)

            for j in range(self.num):
                idx = perm[j, :].reshape(-1, )
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                X = torch.cat([grid_sample, theta_sample], dim=1)

                pwd = sklearn.metrics.pairwise_distances(grid_sample)
                edge_index = np.vstack(np.where(pwd <= self.radius))
                n_edges = edge_index.shape[1]
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                edge_attr = np.zeros((n_edges, 6))
                a = theta_sample[:, 0]
                edge_attr[:, :4] = grid_sample[edge_index.T].reshape(n_edges, -1)
                edge_attr[:, 4] = a[edge_index[0]]
                edge_attr[:, 5] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.num * self.l // batch_size2

        out = torch.zeros(self.n, )
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i].reshape(batch_size2, self.m)
            for j in range(batch_size2):
                pred_ij = pred_i[j, :].reshape(-1, )
                idx = split_idx_i[j, :].reshape(-1, )
                out[idx] = pred_ij

        out = out / self.l

        # out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        # out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1, )


class DownsampleGridSplitter(object):
    def __init__(self, grid, resolution, r, m=100, radius=0.15, edge_features=1):
        super(DownsampleGridSplitter, self).__init__()

        self.grid = grid.reshape(resolution, resolution, 2)
        # self.theta = theta.reshape(resolution, resolution,-1)
        # self.y = y.reshape(resolution, resolution,1)
        self.resolution = resolution
        if resolution % 2 == 1:
            self.s = int(((resolution - 1) / r) + 1)
        else:
            self.s = int(resolution / r)
        self.r = r
        self.n = resolution ** 2
        self.m = m
        self.radius = radius
        self.edge_features = edge_features

        self.index = torch.tensor(range(self.n), dtype=torch.long).reshape(self.resolution, self.resolution)

    def ball_connectivity(self, grid):
        pwd = sklearn.metrics.pairwise_distances(grid)
        edge_index = np.vstack(np.where(pwd <= self.radius))
        n_edges = edge_index.shape[1]
        return torch.tensor(edge_index, dtype=torch.long), n_edges

    def get_data(self, theta):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        data = []
        for x in range(self.r):
            for y in range(self.r):
                grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
                theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)

                perm = torch.randperm(self.n)
                m = self.m - grid_sub.shape[0]
                idx = perm[:m]
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                grid_split = torch.cat([grid_sub, grid_sample], dim=0)
                theta_split = torch.cat([theta_sub, theta_sample], dim=0)
                X = torch.cat([grid_split, theta_split], dim=1)

                edge_index, n_edges = self.ball_connectivity(grid_split)

                edge_attr = np.zeros((n_edges, 4 + self.edge_features * 2))
                a = theta_split[:, :self.edge_features]
                edge_attr[:, :4] = grid_split[edge_index.T].reshape(n_edges, -1)
                edge_attr[:, 4:4 + self.edge_features] = a[edge_index[0]]
                edge_attr[:, 4 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def sample(self, theta, Y):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.resolution, self.resolution)

        x = torch.randint(0, self.r, (1,))
        y = torch.randint(0, self.r, (1,))

        grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
        theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)
        Y_sub = Y[x::self.r, y::self.r].reshape(-1, )
        index_sub = self.index[x::self.r, y::self.r].reshape(-1, )
        n_sub = Y_sub.shape[0]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(self.n, )[idx]

            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=0).reshape(-1, )
            index_split = torch.cat([index_sub, idx], dim=0).reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(-1, )
            index_split = index_sub.reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        edge_index, n_edges = self.ball_connectivity(grid_split)

        edge_attr = np.zeros((n_edges, 4 + self.edge_features * 2))
        a = theta_split[:, :self.edge_features]
        edge_attr[:, :4] = grid_split[edge_index.T].reshape(n_edges, -1)
        edge_attr[:, 4:4 + self.edge_features] = a[edge_index[0]]
        edge_attr[:, 4 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        data = Data(x=X, y=Y_split, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx,
                    sample_idx=index_split)
        print('train', X.shape, Y_split.shape, edge_index.shape, edge_attr.shape, index_split.shape)

        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.r ** 2 // batch_size2

        out = torch.zeros((self.resolution, self.resolution))
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i]
            for j in range(batch_size2):
                pred_ij = pred_i[j, :]
                x, y = split_idx_i[j]
                if self.resolution % 2 == 1:
                    if x == 0:
                        nx = self.s
                    else:
                        nx = self.s - 1
                    if y == 0:
                        ny = self.s
                    else:
                        ny = self.s - 1
                else:
                    nx = self.s
                    ny = self.s
                # pred_ij = pred_i[idx : idx + nx * ny]
                out[x::self.r, y::self.r] = pred_ij[:nx * ny].reshape(nx, ny)

        out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1, )


class TorusGridSplitter(object):
    def __init__(self, grid, resolution, r, m=100, radius=0.15, edge_features=1):
        super(TorusGridSplitter, self).__init__()

        self.grid = grid.reshape(resolution, resolution, 2)
        # self.theta = theta.reshape(resolution, resolution,-1)
        # self.y = y.reshape(resolution, resolution,1)
        self.resolution = resolution
        if resolution % 2 == 1:
            self.s = int(((resolution - 1) / r) + 1)
        else:
            self.s = int(resolution / r)
        self.r = r
        self.n = resolution ** 2
        self.m = m
        self.radius = radius
        self.edge_features = edge_features

        self.index = torch.tensor(range(self.n), dtype=torch.long).reshape(self.resolution, self.resolution)

    def pairwise_difference(self, grid1, grid2):
        n = grid1.shape[0]
        x1 = grid1[:, 0]
        y1 = grid1[:, 1]
        x2 = grid2[:, 0]
        y2 = grid2[:, 1]

        X1 = np.tile(x1.reshape(n, 1), [1, n])
        X2 = np.tile(x2.reshape(1, n), [n, 1])
        X_diff = X1 - X2
        Y1 = np.tile(y1.reshape(n, 1), [1, n])
        Y2 = np.tile(y2.reshape(1, n), [n, 1])
        Y_diff = Y1 - Y2

        return X_diff, Y_diff

    def torus_connectivity(self, grid):
        pwd0 = sklearn.metrics.pairwise_distances(grid, grid)
        X_diff0, Y_diff0 = self.pairwise_difference(grid, grid)

        grid1 = grid
        grid1[:, 0] = grid[:, 0] + 1
        pwd1 = sklearn.metrics.pairwise_distances(grid, grid1)
        X_diff1, Y_diff1 = self.pairwise_difference(grid, grid1)

        grid2 = grid
        grid2[:, 1] = grid[:, 1] + 1
        pwd2 = sklearn.metrics.pairwise_distances(grid, grid2)
        X_diff2, Y_diff2 = self.pairwise_difference(grid, grid2)

        grid3 = grid
        grid3[:, :] = grid[:, :] + 1
        pwd3 = sklearn.metrics.pairwise_distances(grid, grid3)
        X_diff3, Y_diff3 = self.pairwise_difference(grid, grid3)

        grid4 = grid
        grid4[:, 0] = grid[:, 0] + 1
        grid4[:, 1] = grid[:, 1] - 1
        pwd4 = sklearn.metrics.pairwise_distances(grid, grid4)
        X_diff4, Y_diff4 = self.pairwise_difference(grid, grid4)

        PWD = np.stack([pwd0, pwd1, pwd2, pwd3, pwd4], axis=2)
        X_DIFF = np.stack([X_diff0, X_diff1, X_diff2, X_diff3, X_diff4], axis=2)
        Y_DIFF = np.stack([Y_diff0, Y_diff1, Y_diff2, Y_diff3, Y_diff4], axis=2)
        pwd = np.min(PWD, axis=2)
        pwd_index = np.argmin(PWD, axis=2)
        edge_index = np.vstack(np.where(pwd <= self.radius))
        pwd_index = pwd_index[np.where(pwd <= self.radius)]
        PWD_index = (np.where(pwd <= self.radius)[0], np.where(pwd <= self.radius)[1], pwd_index)
        distance = PWD[PWD_index]
        X_difference = X_DIFF[PWD_index]
        Y_difference = Y_DIFF[PWD_index]
        n_edges = edge_index.shape[1]
        return torch.tensor(edge_index, dtype=torch.long), n_edges, distance, X_difference, Y_difference

    def get_data(self, theta):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        data = []
        for x in range(self.r):
            for y in range(self.r):
                grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
                theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)

                perm = torch.randperm(self.n)
                m = self.m - grid_sub.shape[0]
                idx = perm[:m]
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                grid_split = torch.cat([grid_sub, grid_sample], dim=0)
                theta_split = torch.cat([theta_sub, theta_sample], dim=0)
                X = torch.cat([grid_split, theta_split], dim=1)

                edge_index, n_edges, distance, X_difference, Y_difference = self.torus_connectivity(grid_split)

                edge_attr = np.zeros((n_edges, 3 + self.edge_features * 2))
                a = theta_split[:, :self.edge_features]
                edge_attr[:, 0] = X_difference.reshape(n_edges, )
                edge_attr[:, 1] = Y_difference.reshape(n_edges, )
                edge_attr[:, 2] = distance.reshape(n_edges, )
                edge_attr[:, 3:3 + self.edge_features] = a[edge_index[0]]
                edge_attr[:, 3 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def sample(self, theta, Y, connectivity='ball'):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.resolution, self.resolution)

        x = torch.randint(0, self.r, (1,))
        y = torch.randint(0, self.r, (1,))

        grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
        theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)
        Y_sub = Y[x::self.r, y::self.r].reshape(-1, )
        index_sub = self.index[x::self.r, y::self.r].reshape(-1, )
        n_sub = Y_sub.shape[0]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(self.n, )[idx]

            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=0).reshape(-1, )
            index_split = torch.cat([index_sub, idx], dim=0).reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(-1, )
            index_split = index_sub.reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        edge_index, n_edges, distance, X_difference, Y_difference = self.torus_connectivity(grid_split)

        edge_attr = np.zeros((n_edges, 3 + self.edge_features * 2))
        a = theta_split[:, :self.edge_features]
        edge_attr[:, 0] = X_difference.reshape(n_edges, )
        edge_attr[:, 1] = Y_difference.reshape(n_edges, )
        edge_attr[:, 2] = distance.reshape(n_edges, )
        edge_attr[:, 3:3 + self.edge_features] = a[edge_index[0]]
        edge_attr[:, 3 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        data = Data(x=X, y=Y_split, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx,
                    sample_idx=index_split)
        print('train', X.shape, Y_split.shape, edge_index.shape, edge_attr.shape, index_split.shape)

        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.r ** 2 // batch_size2

        out = torch.zeros((self.resolution, self.resolution))
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i]
            for j in range(batch_size2):
                pred_ij = pred_i[j, :]
                x, y = split_idx_i[j]
                if self.resolution % 2 == 1:
                    if x == 0:
                        nx = self.s
                    else:
                        nx = self.s - 1
                    if y == 0:
                        ny = self.s
                    else:
                        ny = self.s - 1
                else:
                    nx = self.s
                    ny = self.s
                # pred_ij = pred_i[idx : idx + nx * ny]
                out[x::self.r, y::self.r] = pred_ij[:nx * ny].reshape(nx, ny)

        out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1, )


def downsample(data, grid_size, l):
    data = data.reshape(-1, grid_size, grid_size)
    data = data[:, ::l, ::l]
    data = data.reshape(-1, (grid_size // l) ** 2)
    return data


def grid(n_x, n_y):
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if (x != n_x - 1):
                edge_index.append((i, i + 1))
                edge_attr.append((1, 0, 0))
                edge_index.append((i + 1, i))
                edge_attr.append((-1, 0, 0))

            if (y != n_y - 1):
                edge_index.append((i, i + n_x))
                edge_attr.append((0, 1, 0))
                edge_index.append((i + n_x, i))
                edge_attr.append((0, -1, 0))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge(n_x, n_y, a):
    a = a.reshape(n_x, n_y)
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if (x != n_x - 1):
                d = 1 / n_x
                a1 = a[x, y]
                a2 = a[x + 1, y]
                edge_index.append((i, i + 1))
                edge_attr.append((d, a1, a2))
                edge_index.append((i + 1, i))
                edge_attr.append((d, a2, a1))

            if (y != n_y - 1):
                d = 1 / n_y
                a1 = a[x, y]
                a2 = a[x, y + 1]
                edge_index.append((i, i + n_x))
                edge_attr.append((d, a1, a2))
                edge_index.append((i + n_x, i))
                edge_attr.append((d, a2, a1))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge_aug(n_x, n_y, a):
    a = a.reshape(n_x, n_y)
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if (x != n_x - 1):
                d = 1 / n_x
                a1 = a[x, y]
                a2 = a[x + 1, y]
                edge_index.append((i, i + 1))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i + 1, i))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

            if (y != n_y - 1):
                d = 1 / n_y
                a1 = a[x, y]
                a2 = a[x, y + 1]
                edge_index.append((i, i + n_x))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i + n_x, i))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge_aug_full(n_x, n_y, r, a):
    n = n_x * n_y

    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)

    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []

    for i1 in range(n):
        x1 = grid[i1]
        for i2 in range(n):
            x2 = grid[i2]

            d = np.linalg.norm(x1 - x2)

            if (d <= r):
                a1 = a[i1]
                a2 = a[i2]
                edge_index.append((i1, i2))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i2, i1))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def multi_grid(depth, n_x, n_y, grid, params):
    edge_index_global = []
    edge_attr_global = []
    X_global = []
    num_nodes = 0

    # build connected graph
    for l in range(depth):
        h_x_l = n_x // (2 ** l)
        h_y_l = n_y // (2 ** l)
        n_l = h_x_l * h_y_l

        a = downsample(params, n_x, (2 ** l))
        if grid == 'grid':
            X, edge_index_inner, edge_attr_inner = grid(h_y_l, h_x_l)
        elif grid == 'grid_edge':
            X, edge_index_inner, edge_attr_inner = grid_edge(h_y_l, h_x_l, a)
        elif grid == 'grid_edge_aug':
            X, edge_index_inner, edge_attr_inner = grid_edge(h_y_l, h_x_l, a)

        # update index
        edge_index_inner = edge_index_inner + num_nodes
        edge_index_global.append(edge_index_inner)
        edge_attr_global.append(edge_attr_inner)

        # construct X
        # if (is_high):
        #     X = torch.cat([torch.zeros(n_l, l * 2), X, torch.zeros(n_l, (depth - 1 - l) * 2)], dim=1)
        # else:
        #     X_l = torch.tensor(l, dtype=torch.float).repeat(n_l, 1)
        #     X = torch.cat([X, X_l], dim=1)
        X_global.append(X)

        # construct edges
        index1 = torch.tensor(range(n_l), dtype=torch.long)
        index1 = index1 + num_nodes
        num_nodes += n_l

        # #construct inter-graph edge
        if l != depth - 1:
            index2 = np.array(range(n_l // 4)).reshape(h_x_l // 2, h_y_l // 2)  # torch.repeat is different from numpy
            index2 = index2.repeat(2, axis=0).repeat(2, axis=1)
            index2 = torch.tensor(index2).reshape(-1)
            index2 = index2 + num_nodes
            index2 = torch.tensor(index2, dtype=torch.long)

            edge_index_inter1 = torch.cat([index1, index2], dim=-1).reshape(2, -1)
            edge_index_inter2 = torch.cat([index2, index1], dim=-1).reshape(2, -1)
            edge_index_inter = torch.cat([edge_index_inter1, edge_index_inter2], dim=1)

            edge_attr_inter1 = torch.tensor((0, 0, 1), dtype=torch.float).repeat(n_l, 1)
            edge_attr_inter2 = torch.tensor((0, 0, -1), dtype=torch.float).repeat(n_l, 1)
            edge_attr_inter = torch.cat([edge_attr_inter1, edge_attr_inter2], dim=0)

            edge_index_global.append(edge_index_inter)
            edge_attr_global.append(edge_attr_inter)

    X = torch.cat(X_global, dim=0)
    edge_index = torch.cat(edge_index_global, dim=1)
    edge_attr = torch.cat(edge_attr_global, dim=0)
    mask_index = torch.tensor(range(n_x * n_y), dtype=torch.long)
    # print('create multi_grid with size:', X.shape,  edge_index.shape, edge_attr.shape, mask_index.shape)

    return (X, edge_index, edge_attr, mask_index, num_nodes)

class OBMeshfree(nn.Module):
    def __init__(self, delta, x, ID, batch_size):
        super(OBMeshfree, self).__init__()
        self.basedim=21 #21
        self.delta=delta
        self.x=x
        self.ID=ID
        self.batch_size=batch_size

    def neighborlist(self):
        delta=self.delta
        x=self.x
        N = len(x)
        nei = [[] for _ in range(N)]
        for i in range(N): #loop over all the particle
            #time0 = time.time()
            for j in range(N): #searching neighborhood for each particle
                r=np.sqrt((x[j][0]-x[i][0])**2+(x[j][1]-x[i][1])**2)
                if (r<=delta+1e-8):
                    nei[i].append(j)
            #print(time.time()-time0)
        self.nei=nei
        return nei

    #@staticmethod
    def Phi(self, x,y):
        basedim=self.basedim
        phi=np.zeros((basedim))
        phi[0] = 1.0
        phi[1] = x
        phi[2] = y
        phi[3] = x*x
        phi[4] = x*y
        phi[5] = y*y
        phi[6] = x*x*x
        phi[7] = x*x*y
        phi[8] = x*y*y
        phi[9] = y*y*y
        phi[10] = x*x*x*x
        phi[11] = x*x*x*y
        phi[12] = x*x*y*y
        phi[13] = x*y*y*y
        phi[14] = y*y*y*y
        phi[15] = x*x*x*x*x
        phi[16] = x*x*x*x*y
        phi[17] = x*x*x*y*y
        phi[18] = x*x*y*y*y
        phi[19] = x*y*y*y*y
        phi[20] = y*y*y*y*y
        return phi

    #@staticmethod
    def IPhi_dxy(self, x,y, delta):
        # for integral
        basedim=self.basedim
        IPhi=np.zeros((basedim))
        IPhi[4] = 1.0
        IPhi[7] = 2*x
        IPhi[8] = 2*y
        IPhi[11] = 3*x**2
        IPhi[12] = 4*x*y
        IPhi[13] = 3*y**2
        IPhi[16] = 4*x**3
        IPhi[17] = 6*x**2*y
        IPhi[18] = 6*x*y**2
        IPhi[19] = 4*y**3
        return IPhi

    def IPhi_dyy(self, x,y, delta):
        # for integral
        basedim=self.basedim
        IPhi=np.zeros((basedim))
        IPhi[5] = 2.0
        IPhi[8] = 2*x
        IPhi[9] = 6*y
        IPhi[12] = 2*x**2
        IPhi[13] = 6*x*y
        IPhi[14] = 12*y**2
        IPhi[17] = 2*x**3
        IPhi[18] = 6*x**2*y
        IPhi[19] = 12*x*y**2
        IPhi[20] = 20*y**3
        return IPhi

    def IPhi_dxx(self, x,y, delta):
        # for integral
        basedim=self.basedim
        IPhi=np.zeros((basedim))
        IPhi[3] = 2.0
        IPhi[6] = 6*x
        IPhi[7] = 2*y
        IPhi[10] = 12*x**2
        IPhi[11] = 6*x*y
        IPhi[12] = 2*y**2
        IPhi[15] = 20*x**3
        IPhi[16] = 12*x**2*y
        IPhi[17] = 2*y**3
        IPhi[18] = 2*y**3
        return IPhi

    def IPhi_int(self, x,y, delta):
        # for integral
        basedim=self.basedim
        IPhi=np.zeros((basedim))
        IPhi[0]=math.pi*delta**2
        IPhi[3]=1.0/4.0*math.pi*delta**4
        IPhi[5] = 1.0 / 4.0 * math.pi * delta ** 4
        IPhi[10] = 1.0 / 8.0 * math.pi * delta ** 6
        IPhi[12] = 1.0 / 24.0 * math.pi * delta ** 6
        IPhi[14] = 1.0 / 8.0 * math.pi * delta ** 6
        return IPhi

    def IPhi_dx(self, x,y, delta):
        # for du/dx
        basedim=self.basedim
        IPhi=np.zeros((basedim))
        IPhi[0]=0.0
        IPhi[1]=1.0
        IPhi[2] = 0.0
        IPhi[3] = 2.0*x
        IPhi[4] = y
        IPhi[5] = 0.0
        IPhi[6] = 3.0 * x**2
        IPhi[7] = 2.0 * x * y
        IPhi[8] = y**2
        IPhi[9] = 0.0
        IPhi[10] = 4 * x**3
        IPhi[11] = 3 * x**2 * y
        IPhi[12] = 2.0 * x * y**2
        IPhi[13] = y**3
        IPhi[14] = 0.0
        IPhi[15] = 5 * x**4
        IPhi[16] = 4 * x**3 * y
        IPhi[17] = 3 * x**2 * y**2
        IPhi[18] = 2 * x * y**3
        IPhi[19] = y**4
        IPhi[20] = 0.0
        return IPhi

    def IPhi_dy(self, x,y, delta):
        basedim=self.basedim
        IPhi=np.zeros((basedim))
        IPhi[0]=0.0
        IPhi[1]=0.0
        IPhi[2] = 1.0
        IPhi[3] = 0.0
        IPhi[4] = x
        IPhi[5] = 2*y
        IPhi[6] = 0.0
        IPhi[7] = x**2
        IPhi[8] = 2.0 * y * x
        IPhi[9] = 3.0 * y**2
        IPhi[10] = 0.0
        IPhi[11] = x**3
        IPhi[12] = 2 * x**2 * y
        IPhi[13] = 3 * x * y**2
        IPhi[14] = 4 * y**3
        IPhi[15] = 0.0
        IPhi[16] = x**4
        IPhi[17] = 2 * x**3 * y
        IPhi[18] = 3 * x**2 * y**2
        IPhi[19] = 4 * x * y**3
        IPhi[20] = 5 * y**4
        return IPhi

    def IPhi_dx3(self, x,y, delta):
        basedim=self.basedim
        IPhi=np.zeros((basedim))
        IPhi[0]=0.0
        IPhi[1]=0.0
        IPhi[2] = 0.0
        IPhi[3] = 0.0
        IPhi[4] = 0.0
        IPhi[5] = 0.0
        IPhi[6] = 6.0
        IPhi[7] = 0.0
        IPhi[8] = 0.0
        IPhi[9] = 0.0
        IPhi[10] = 24 * x
        IPhi[11] = 6 * y
        IPhi[12] = 0.0
        IPhi[13] = 0.0
        IPhi[14] = 0.0
        IPhi[15] = 60 * x**2
        IPhi[16] = 24 * x * y
        IPhi[17] = 6 * y**2
        IPhi[18] = 0.0
        IPhi[19] = 0.0
        IPhi[20] = 0.0
        return IPhi

    def IPhi_dx2y(self, x,y, delta):
        basedim=self.basedim
        IPhi=np.zeros((basedim))
        IPhi[0]=0.0
        IPhi[1]=0.0
        IPhi[2] = 0.0
        IPhi[3] = 0.0
        IPhi[4] = 0.0
        IPhi[5] = 0.0
        IPhi[6] = 0.0
        IPhi[7] = 2.0
        IPhi[8] = 0.0
        IPhi[9] = 0.0
        IPhi[10] = 0.0
        IPhi[11] = 6 * x
        IPhi[12] = 4 * y
        IPhi[13] = 0.0
        IPhi[14] = 0.0
        IPhi[15] = 0.0
        IPhi[16] = 12 * x**2
        IPhi[17] = 12 * x * y
        IPhi[18] = 6 * y**2
        IPhi[19] = 0.0
        IPhi[20] = 0.0
        return IPhi

    def IPhi_dxy2(self, x,y, delta):
        basedim=self.basedim
        IPhi=np.zeros((basedim))
        IPhi[0]=0.0
        IPhi[1]=0.0
        IPhi[2] = 0.0
        IPhi[3] = 0.0
        IPhi[4] = 0.0
        IPhi[5] = 0.0
        IPhi[6] = 0.0
        IPhi[7] = 0.0
        IPhi[8] = 2.0
        IPhi[9] = 0.0
        IPhi[10] = 0.0
        IPhi[11] = 0.0
        IPhi[12] = 4 * x
        IPhi[13] = 6 * y
        IPhi[14] = 0.0
        IPhi[15] = 0.0
        IPhi[16] = 0.0
        IPhi[17] = 6 * x**2
        IPhi[18] = 12 * x * y
        IPhi[19] = 12 * y**3
        IPhi[20] = 0.0
        return IPhi

    def IPhi_dy3(self, x,y, delta):
        basedim=self.basedim
        IPhi=np.zeros((basedim))
        IPhi[0]=0.0
        IPhi[1]=0.0
        IPhi[2] = 0.0
        IPhi[3] = 0.0
        IPhi[4] = 0.0
        IPhi[5] = 0.0
        IPhi[6] = 0.0
        IPhi[7] = 0.0
        IPhi[8] = 0.0
        IPhi[9] = 6.0
        IPhi[10] = 0.0
        IPhi[11] = 0.0
        IPhi[12] = 0.0
        IPhi[13] = 6 * x
        IPhi[14] = 24 * y
        IPhi[15] = 0.0
        IPhi[16] = 0.0
        IPhi[17] = 0.0
        IPhi[18] = 6 * x**2
        IPhi[19] = 24 * x * y
        IPhi[20] = 60 * y**2
        return IPhi

    def quad_weights(self, cur_neighborhood, ind, case):
        N=len(cur_neighborhood)
        x=self.x
        delta=self.delta
        basedim=self.basedim
        B=np.zeros((basedim,N))
        #weights=np.zeros((N))
        for i in range(N):
            B[:,i]= self.Phi((x[cur_neighborhood[i]][0] - x[ind][0])/delta, (x[cur_neighborhood[i]][1] - x[ind][1])/delta)

        oldB = np.zeros((basedim, N))
        for i in range(N):
            oldB[:,i]= self.Phi((x[cur_neighborhood[i]][0] - x[ind][0]), (x[cur_neighborhood[i]][1] - x[ind][1]))

        if (case=='dx'):
            g = self.IPhi_dx(0.0, 0.0, delta)/delta
            M = np.matmul(B, B.transpose())
            weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M), g))
        elif (case=='dy'):
            g = self.IPhi_dy(0.0, 0.0, delta)/delta
            M = np.matmul(B, B.transpose())
            weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M), g))
        elif (case=='dx3'):
            g = self.IPhi_dx3(0.0, 0.0, delta)/delta**3
            M = np.matmul(B, B.transpose())
            weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M), g))
        elif (case=='dx2y'):
            g = self.IPhi_dx2y(0.0, 0.0, delta)/delta**3
            M = np.matmul(B, B.transpose())
            weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M), g))
        elif (case=='dxy2'):
            g = self.IPhi_dxy2(0.0, 0.0, delta)/delta**3
            M = np.matmul(B, B.transpose())
            weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M), g))
        elif (case=='dy3'):
            g = self.IPhi_dy3(0.0, 0.0, delta)/delta**3
            M = np.matmul(B, B.transpose())
            weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M), g))
        else:
            print('Wrong case!')

        # M = np.matmul(B, B.transpose())
        # weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M), g))

        return weights

    def get_all_weights(self):
        x=self.x
        neighborhood=self.neighborlist()
        N=len(x)
        all_weights_dx=[[] for _ in range(N)]
        all_weights_dy=[[] for _ in range(N)]

        for i in range(N):
            # if (self.ID[i]==0):
            #     print('test!')
            #time0=time.time()
            all_weights_dx[i] = self.quad_weights(neighborhood[i], i, 'dx')
            all_weights_dy[i] = self.quad_weights(neighborhood[i], i, 'dy')
            #print(time.time()-time0)

        return all_weights_dx, all_weights_dy

    def build_matrix(self):
        x = self.x
        nei = self.neighborlist()
        wei_dx, wei_dy=self.get_all_weights()
        N = len(x)
        sm_dx = np.zeros((N,N))
        sm_dy = np.zeros((N,N))
        for i in range(N):
            sm_dx[i, nei[i]] = wei_dx[i]
            sm_dy[i, nei[i]] = wei_dy[i]

        return scipy.linalg.block_diag(*[sm_dx for i in range(self.batch_size)]), scipy.linalg.block_diag(*[sm_dy for i in range(self.batch_size)])

    def get_all_weights_d3(self):
        x=self.x
        neighborhood=self.neighborlist()
        N=len(x)
        all_weights_dx3 = [[] for _ in range(N)]
        all_weights_dx2y = [[] for _ in range(N)]
        all_weights_dxy2 = [[] for _ in range(N)]
        all_weights_dy3 = [[] for _ in range(N)]
        for i in range(N):
            #if (self.ID[i]==0):
                #print('test!')
            all_weights_dx3[i] = self.quad_weights(neighborhood[i], i, 'dx3')
            all_weights_dx2y[i] = self.quad_weights(neighborhood[i], i, 'dx2y')
            all_weights_dxy2[i] = self.quad_weights(neighborhood[i], i, 'dxy2')
            all_weights_dy3[i] = self.quad_weights(neighborhood[i], i, 'dy3')

        return all_weights_dx3, all_weights_dx2y, all_weights_dxy2, all_weights_dy3

    def build_matrix_d3(self):
        x = self.x
        nei = self.neighborlist()
        wei_dx3, wei_dx2y, wei_dxy2, wei_dy3=self.get_all_weights_d3()
        N = len(x)
        sm_dx3 = np.zeros((N,N))
        sm_dx2y = np.zeros((N,N))
        sm_dxy2 = np.zeros((N,N))
        sm_dy3 = np.zeros((N,N))
        for i in range(N):
            sm_dx3[i, nei[i]] = wei_dx3[i]
            sm_dx2y[i, nei[i]] = wei_dx2y[i]
            sm_dxy2[i, nei[i]] = wei_dxy2[i]
            sm_dy3[i, nei[i]] = wei_dy3[i]

        return scipy.linalg.block_diag(*[sm_dx3 for i in range(self.batch_size)]), scipy.linalg.block_diag(*[sm_dx2y for i in range(self.batch_size)]), scipy.linalg.block_diag(*[sm_dxy2 for i in range(self.batch_size)]), scipy.linalg.block_diag(*[sm_dy3 for i in range(self.batch_size)])

    def OBM_get_int_hor_and_weights(self, x, ID):

        delta=self.delta
        basedim=self.basedim
        N1=len(x)

        ID1=ID
        # find neighbor list for new domain
        hor = [[] for _ in range(N1)]
        for i in range(N1):  # loop over all the particle
            for j in range(N1):  # searching neighborhood for each particle
                r = np.sqrt((x[j][0]-x[i][0])**2 + (x[j][1]-x[i][1])**2)
                if (r <= delta):
                    hor[i].append(j)

        #get all weights for all particles
        all_weights_int = [[] for _ in range(N1)]
        for ind in range(N1):
            if ID1[ind]!=2:
                cur_horizon=hor[ind]
                nh = len(cur_horizon)
                B = np.zeros((basedim, nh))
                for i in range(nh):
                    B[:, i] = self.Phi(x[cur_horizon[i]][0] - x[ind][0], x[cur_horizon[i]][1] - x[ind][1])
                g = self.IPhi_int(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_int[ind] = weights

        return hor, all_weights_int

    def OBM_diffusion_solver(self, f, x, ID, hor, all_weights_int, uxapp, uyapp):
        # solver for static diffusion, Neumann bd=0, rhs(loading term f)
        #x=self.x
        delta=self.delta
        basedim=self.basedim
        N1=len(x)

        ID1=ID

        interpx = LinearNDInterpolator(x, uxapp)
        interpy = LinearNDInterpolator(x, uyapp)

        sm_diff = np.zeros((N1, N1))
        rhs = np.zeros((N1))
        for i in range(N1):
            if ID1[i]==0: #inner  particles
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                rhs[i] = f[i]
                for j in range(nh):
                    sm_diff[i, cur_horizon[j]] += (-2 * 4/math.pi/delta**4) * all_weights_int[i][j]
                    sm_diff[i, i] -= (-2 * 4/math.pi/delta ** 4) * all_weights_int[i][j]
            elif ID1[i]==1: # parts of the horizon is outside the domain
                # if i>52:
                #     print('zaizheintgdun')
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                xn=math.sqrt(x[i][0]**2+x[i][1]**2)
                normal_x = x[i][0]/xn
                normal_y = x[i][1]/xn
                x_bar = 0.4*normal_x
                y_bar = 0.4*normal_y
                uxbar=interpx(x_bar,y_bar)
                uybar=interpy(x_bar,y_bar)
                rhs[i] = f[i]
                for j in range(nh):
                    #g_xbar = (math.cos(x_bar) * math.cos(y_bar)) * normal_x +(-math.sin(x_bar) * math.sin(y_bar)) * normal_y
                    #g_xbar = 1 * normal_x +1 * normal_y
                    #g_xbar = (3 * x_bar ** 2) * normal_x + (0.0) * normal_y
                    g_xbar=0.0
                    g_xbar=uxbar * normal_x + uybar * normal_y
                    if ID1[cur_horizon[j]]!=2: # for the particles inside the domain (including Neumann bd)
                        sm_diff[i,cur_horizon[j]] += (-2 * 4/math.pi/delta**4) * all_weights_int[i][j]
                        sm_diff[i, i] -= (-2 * 4/math.pi/delta**4) * all_weights_int[i][j]
                    else:
                        #rhs[i] -= (-2 * 4/math.pi/delta**4 * ((x[cur_horizon[j]][0]-x[i][0])*(normal_x)+(x[cur_horizon[j]][1]-x[i][1])*(normal_y))) * all_weights_int[i][j] * 0.0 #du/dn=g=0
                        rhs[i] -= (-2 * 4/math.pi/delta**4 * ((x[cur_horizon[j]][0]-x[i][0])*(normal_x)+(x[cur_horizon[j]][1]-x[i][1])*(normal_y))) * all_weights_int[i][j] * g_xbar  # du/dn=g
        result=np.matmul(np.linalg.pinv(sm_diff[:,ID1!=2][ID1!=2]),rhs[ID1!=2])
        return result

    def OBM_diffusion_solver_diffmatrix(self, x, ID, hor, all_weights_int):
        delta=self.delta
        N1=len(x)

        sm_diff = np.zeros((N1, N1))
        rhs = np.zeros((N1))
        for i in range(N1):
            if ID[i]==0: #inner  particles
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                for j in range(nh):
                    sm_diff[i, cur_horizon[j]] += (-2 * 4/math.pi/delta**4) * all_weights_int[i][j]
                    sm_diff[i, i] -= (-2 * 4/math.pi/delta ** 4) * all_weights_int[i][j]
            elif ID[i]==1: # parts of the horizon is outside the domain
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                for j in range(nh):
                    if ID[cur_horizon[j]]!=2: # for the particles inside the domain (including Neumann bd)
                        sm_diff[i,cur_horizon[j]] += (-2 * 4/math.pi/delta**4) * all_weights_int[i][j]
                        sm_diff[i, i] -= (-2 * 4/math.pi/delta**4) * all_weights_int[i][j]
        return sm_diff[:,ID!=2][ID!=2]

    def OBM_diffusion_solver_newrhs(self, f, x, interx, ID, hor, all_weights_int, uxapp, uyapp):
        # solver for static diffusion, Neumann bd=0, rhs(loading term f)
        #x=self.x
        delta=self.delta
        N1=len(x)

        # interpx = LinearNDInterpolator(interx, uxapp)
        # interpy = LinearNDInterpolator(interx, uyapp)

        rhs = np.zeros((N1))
        for i in range(N1):
            # if i==65:
            #     print('stop and wait a minute')
            if ID[i]==0: #inner  particles
                rhs[i] = f[i]
            elif ID[i]==1: # parts of the horizon is outside the domain
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                xn=math.sqrt(x[i][0]**2+x[i][1]**2)
                normal_x = x[i][0]/xn
                normal_y = x[i][1]/xn
                x_bar = 0.4*normal_x
                y_bar = 0.4*normal_y
                #uxbar=interpx(x_bar*(1-1e-2),y_bar*(1-1e-2)) # add a tolerance since interp will blow up for outside points
                #uybar=interpy(x_bar*(1-1e-2),y_bar*(1-1e-2))
                uxbar = 3*x_bar**2  # add a tolerance since interp will blow up for outside points
                uybar = 3*y_bar**2
                uxbar = 0.0
                uybar = 0.0
                rhs[i] = f[i]
                for j in range(nh):
                    g_xbar=uxbar * normal_x + uybar * normal_y
                    if ID[cur_horizon[j]] == 2:
                        #rhs[i] -= (-2 * 4/math.pi/delta**4 * ((x[cur_horizon[j]][0]-x[i][0])*(normal_x)+(x[cur_horizon[j]][1]-x[i][1])*(normal_y))) * all_weights_int[i][j] * 0.0 #du/dn=g=0
                        rhs[i] -= (-2 * 4/math.pi/delta**4 * ((x[cur_horizon[j]][0]-x[i][0])*(normal_x)+(x[cur_horizon[j]][1]-x[i][1])*(normal_y))) * all_weights_int[i][j] * g_xbar  # du/dn=g
        return rhs[ID!=2]

    def OBM_diffusion_solver_2nd(self, f, x, ID):
        # solver for static diffusion, Neumann bd=0, rhs(loading term f)
        #x = self.x
        delta = self.delta
        basedim = self.basedim
        N1 = len(x)

        ID1=ID
        # find neighbor list for new domain
        hor = [[] for _ in range(N1)]
        for i in range(N1):  # loop over all the particle
            for j in range(N1):  # searching neighborhood for each particle
                r = np.sqrt((x[j][0] - x[i][0]) ** 2 + (x[j][1] - x[i][1]) ** 2)
                if (r <= delta):
                    hor[i].append(j)

        # find neighbor list for old domain
        nei = [[] for _ in range(N1)]
        for i in range(N1):  # loop over all the particle
            for j in range(N1):  # searching neighborhood for each particle
                r = np.sqrt((x[j][0] - x[i][0]) ** 2 + (x[j][1] - x[i][1]) ** 2)
                if (r <= delta and ID1[j] != 2):
                    nei[i].append(j)

        # get all weights for all particles
        all_weights_int = [[] for _ in range(N1)]
        all_weights_dxx = [[] for _ in range(N1)]
        all_weights_dyy = [[] for _ in range(N1)]
        all_weights_dxy = [[] for _ in range(N1)]
        for ind in range(N1):
            if ID1[ind] != 2:
                cur_horizon = hor[ind]
                nh = len(cur_horizon)
                B = np.zeros((basedim, nh))
                for i in range(nh):
                    B[:, i] = self.Phi(x[cur_horizon[i]][0] - x[ind][0], x[cur_horizon[i]][1] - x[ind][1])

                g = self.IPhi_int(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_int[ind] = weights

        for ind in range(N1):
            if ID1[ind] != 2:
                cur_nei = nei[ind]
                nh = len(cur_nei)
                B = np.zeros((basedim, nh))
                for i in range(nh):
                    B[:, i] = self.Phi(x[cur_nei[i]][0] - x[ind][0], x[cur_nei[i]][1] - x[ind][1])

                g = self.IPhi_dxx(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_dxx[ind] = weights

                g = self.IPhi_dyy(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_dyy[ind] = weights

                g = self.IPhi_dxy(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_dxy[ind] = weights

        sm_diff = np.zeros((N1, N1))
        rhs = np.zeros((N1))
        for i in range(N1):
            if ID1[i] == 0:  # inner  particles
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                rhs[i] = f[i]
                for j in range(nh):
                    sm_diff[i, cur_horizon[j]] += (-2 * 4 / math.pi / delta ** 4) * all_weights_int[i][j]
                    sm_diff[i, i] -= (-2 * 4 / math.pi / delta ** 4) * all_weights_int[i][j]
            elif ID1[i] == 1:  # parts of the horizon is outside the domain
                # if i>52:
                #     print('zaizheintgdun')
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                xn = math.sqrt(x[i][0] ** 2 + x[i][1] ** 2)
                normal_x = x[i][0] / xn
                normal_y = x[i][1] / xn
                tanvec_x = -x[i][1] / xn
                tanvec_y = x[i][0] / xn
                x_bar = 0.4 * normal_x
                y_bar = 0.4 * normal_y
                rhs[i] = f[i]

                # compute M_delta(\xb)
                Md=0.0
                for j in range(nh):
                    if ID1[cur_horizon[j]] == 2:
                        #\int ((y-x)p)^2-((y-x_bar)n)^2+((x-x_bar))^2
                        Md+=(4/math.pi/delta**4)*(((x[cur_horizon[j]][0]-x[i][0])*tanvec_x+(x[cur_horizon[j]][1]-x[i][1])*tanvec_y)**2-((x[cur_horizon[j]][0]-x_bar)*normal_x+(x[cur_horizon[j]][1]-y_bar)*normal_y)**2+((x[i][0]-x_bar)*normal_x+(x[i][1]-y_bar)*normal_y)**2)*all_weights_int[i][j]

                for j in range(nh):
                    #g_xbar=(1)*normal_x+(1)*normal_y
                    g_xbar = (math.cos(x_bar) * math.cos(y_bar)) * normal_x + (-math.sin(x_bar) * math.sin(y_bar)) * normal_y
                    g_xbar = (3*x_bar**2) * normal_x + (0.0) * normal_y
                    g_xbar=0.0
                    if ID1[cur_horizon[j]] != 2:  # for the particles inside the domain (including Neumann bd)
                        # part 1: nonlocal integral
                        sm_diff[i, cur_horizon[j]] += (-2*4/math.pi/delta**4) * all_weights_int[i][j]
                        sm_diff[i, i] -= (-2*4/math.pi/delta**4) * all_weights_int[i][j]
                    else:
                        rhs[i] += (2 * 4/math.pi/delta**4 * ((x[cur_horizon[j]][0]-x[i][0])*normal_x+(x[cur_horizon[j]][1]-x[i][1])*normal_y)) * all_weights_int[i][j] * g_xbar  # du/dn=g
                        rhs[i] -= (4/math.pi/delta**4)*(((x[cur_horizon[j]][0]-x_bar)*normal_x+(x[cur_horizon[j]][1]-y_bar)*normal_y)**2-((x[i][0]-x_bar)*normal_x+(x[i][1]-y_bar)*normal_y)**2)*all_weights_int[i][j] * f[i]
                cur_nei = nei[i]
                nh = len(cur_nei)
                for j in range(nh):
                    # part 2: u(x)_pp
                    sm_diff[i, cur_nei[j]] += -Md * (tanvec_x ** 2 * all_weights_dxx[i][j] + 2 * tanvec_x * tanvec_y * all_weights_dxy[i][j] + tanvec_y ** 2 * all_weights_dyy[i][j])
                    sm_diff[i, i] -= -Md * (tanvec_x ** 2 * all_weights_dxx[i][j] + 2 * tanvec_x * tanvec_y * all_weights_dxy[i][j] + tanvec_y ** 2 * all_weights_dyy[i][j])
        result = np.matmul(np.linalg.pinv(sm_diff[:, ID1!=2][ID1!=2]), rhs[ID1 != 2])
        return result

    def OBM_get_int_hor_and_weights_2nd(self, x, ID):

        delta = self.delta
        basedim = self.basedim
        N1 = len(x)

        ID1 = ID
        # find neighbor list for new domain
        hor = [[] for _ in range(N1)]
        for i in range(N1):  # loop over all the particle
            for j in range(N1):  # searching neighborhood for each particle
                r = np.sqrt((x[j][0] - x[i][0]) ** 2 + (x[j][1] - x[i][1]) ** 2)
                if (r <= delta):
                    hor[i].append(j)

        # find neighbor list for old domain
        nei = [[] for _ in range(N1)]
        for i in range(N1):  # loop over all the particle
            for j in range(N1):  # searching neighborhood for each particle
                r = np.sqrt((x[j][0] - x[i][0]) ** 2 + (x[j][1] - x[i][1]) ** 2)
                if (r <= delta and ID1[j] != 2):
                    nei[i].append(j)

        # get all weights for all particles
        all_weights_int = [[] for _ in range(N1)]
        all_weights_dxx = [[] for _ in range(N1)]
        all_weights_dyy = [[] for _ in range(N1)]
        all_weights_dxy = [[] for _ in range(N1)]
        for ind in range(N1):
            if ID1[ind] != 2:
                cur_horizon = hor[ind]
                nh = len(cur_horizon)
                B = np.zeros((basedim, nh))
                for i in range(nh):
                    B[:, i] = self.Phi(x[cur_horizon[i]][0] - x[ind][0], x[cur_horizon[i]][1] - x[ind][1])

                g = self.IPhi_int(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_int[ind] = weights

        for ind in range(N1):
            if ID1[ind] != 2:
                cur_nei = nei[ind]
                nh = len(cur_nei)
                B = np.zeros((basedim, nh))
                for i in range(nh):
                    B[:, i] = self.Phi(x[cur_nei[i]][0] - x[ind][0], x[cur_nei[i]][1] - x[ind][1])

                g = self.IPhi_dxx(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_dxx[ind] = weights

                g = self.IPhi_dyy(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_dyy[ind] = weights

                g = self.IPhi_dxy(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_dxy[ind] = weights

        return hor, nei, all_weights_int, all_weights_dxx, all_weights_dxy, all_weights_dyy

    def OBM_diffusion_solver_2nd_diffmatrix(self, x, ID, hor, nei, all_weights_int, all_weights_dxx, all_weights_dxy, all_weights_dyy):
        # solver for static diffusion, Neumann bd=0, rhs(loading term f)
        #x = self.x
        delta = self.delta
        basedim = self.basedim
        N1 = len(x)

        sm_diff = np.zeros((N1, N1))
        #rhs = np.zeros((N1))
        for i in range(N1):
            if ID[i] == 0:  # inner  particles
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                #rhs[i] = f[i]
                for j in range(nh):
                    sm_diff[i, cur_horizon[j]] += (-2 * 4 / math.pi / delta ** 4) * all_weights_int[i][j]
                    sm_diff[i, i] -= (-2 * 4 / math.pi / delta ** 4) * all_weights_int[i][j]
            elif ID[i] == 1:  # parts of the horizon is outside the domain
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                xn = math.sqrt(x[i][0] ** 2 + x[i][1] ** 2)
                normal_x = x[i][0] / xn
                normal_y = x[i][1] / xn
                tanvec_x = -x[i][1] / xn
                tanvec_y = x[i][0] / xn
                x_bar = 0.4 * normal_x
                y_bar = 0.4 * normal_y
                #rhs[i] = f[i]

                # compute M_delta(\xb)
                Md=0.0
                for j in range(nh):
                    if ID[cur_horizon[j]] == 2:
                        #\int ((y-x)p)^2-((y-x_bar)n)^2+((x-x_bar))^2
                        Md+=(4/math.pi/delta**4)*(((x[cur_horizon[j]][0]-x[i][0])*tanvec_x+(x[cur_horizon[j]][1]-x[i][1])*tanvec_y)**2-((x[cur_horizon[j]][0]-x_bar)*normal_x+(x[cur_horizon[j]][1]-y_bar)*normal_y)**2+((x[i][0]-x_bar)*normal_x+(x[i][1]-y_bar)*normal_y)**2)*all_weights_int[i][j]

                for j in range(nh):
                    if ID[cur_horizon[j]] != 2:  # for the particles inside the domain (including Neumann bd)
                        # part 1: nonlocal integral
                        sm_diff[i, cur_horizon[j]] += (-2*4/math.pi/delta**4) * all_weights_int[i][j]
                        sm_diff[i, i] -= (-2*4/math.pi/delta**4) * all_weights_int[i][j]
                    # else:
                    #     rhs[i] += (2 * 4/math.pi/delta**4 * ((x[cur_horizon[j]][0]-x[i][0])*normal_x+(x[cur_horizon[j]][1]-x[i][1])*normal_y)) * all_weights_int[i][j] * g_xbar  # du/dn=g
                    #     rhs[i] -= (4/math.pi/delta**4)*(((x[cur_horizon[j]][0]-x_bar)*normal_x+(x[cur_horizon[j]][1]-y_bar)*normal_y)**2-((x[i][0]-x_bar)*normal_x+(x[i][1]-y_bar)*normal_y)**2)*all_weights_int[i][j] * f[i]
                cur_nei = nei[i]
                nh = len(cur_nei)
                for j in range(nh):
                    # part 2: u(x)_pp
                    sm_diff[i, cur_nei[j]] += -Md * (tanvec_x ** 2 * all_weights_dxx[i][j] + 2 * tanvec_x * tanvec_y * all_weights_dxy[i][j] + tanvec_y ** 2 * all_weights_dyy[i][j])
                    sm_diff[i, i] -= -Md * (tanvec_x ** 2 * all_weights_dxx[i][j] + 2 * tanvec_x * tanvec_y * all_weights_dxy[i][j] + tanvec_y ** 2 * all_weights_dyy[i][j])
        return sm_diff[:,ID!=2][ID!=2]


    def OBM_diffusion_solver_2nd_rhs(self, f, x, ID, hor, nei, all_weights_int, all_weights_dxx, all_weights_dxy, all_weights_dyy):
        # solver for static diffusion, Neumann bd=0, rhs(loading term f)
        #x = self.x
        delta = self.delta
        basedim = self.basedim
        N1 = len(x)

        sm_diff = np.zeros((N1, N1))
        rhs = np.zeros((N1))
        for i in range(N1):
            if ID[i] == 0:  # inner  particles
                cur_horizon = hor[i]
                rhs[i] = f[i]
            elif ID[i] == 1:  # parts of the horizon is outside the domain
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                xn = math.sqrt(x[i][0] ** 2 + x[i][1] ** 2)
                normal_x = x[i][0] / xn
                normal_y = x[i][1] / xn
                tanvec_x = -x[i][1] / xn
                tanvec_y = x[i][0] / xn
                x_bar = 0.4 * normal_x
                y_bar = 0.4 * normal_y
                rhs[i] = f[i]

                uxbar = 3 * x_bar ** 2  # add a tolerance since interp will blow up for outside points
                uybar = 3 * y_bar ** 2

                uxbar = 0.0
                uybar = 0.0

                # compute M_delta(\xb)
                Md=0.0
                for j in range(nh):
                    if ID[cur_horizon[j]] == 2:
                        #\int ((y-x)p)^2-((y-x_bar)n)^2+((x-x_bar))^2
                        Md+=(4/math.pi/delta**4)*(((x[cur_horizon[j]][0]-x[i][0])*tanvec_x+(x[cur_horizon[j]][1]-x[i][1])*tanvec_y)**2-((x[cur_horizon[j]][0]-x_bar)*normal_x+(x[cur_horizon[j]][1]-y_bar)*normal_y)**2+((x[i][0]-x_bar)*normal_x+(x[i][1]-y_bar)*normal_y)**2)*all_weights_int[i][j]
                for j in range(nh):
                    g_xbar=0.0
                    g_xbar=uxbar*normal_x+uybar*normal_y
                    if ID[cur_horizon[j]] == 2:
                        rhs[i] += (2 * 4/math.pi/delta**4 * ((x[cur_horizon[j]][0]-x[i][0])*normal_x+(x[cur_horizon[j]][1]-x[i][1])*normal_y)) * all_weights_int[i][j] * g_xbar  # du/dn=g
                        rhs[i] -= (4/math.pi/delta**4)*(((x[cur_horizon[j]][0]-x_bar)*normal_x+(x[cur_horizon[j]][1]-y_bar)*normal_y)**2-((x[i][0]-x_bar)*normal_x+(x[i][1]-y_bar)*normal_y)**2)*all_weights_int[i][j] * f[i]
        return rhs[ID!=2]


    def OBM_diffusion_solver_2ndsq(self, f, ID1):
        # solver for static diffusion, Neumann bd=0, rhs(loading term f)
        x = self.x
        delta = self.delta
        basedim = self.basedim
        N1 = len(x)

        # find neighbor list for new domain
        hor = [[] for _ in range(N1)]
        for i in range(N1):  # loop over all the particle
            for j in range(N1):  # searching neighborhood for each particle
                r = np.sqrt((x[j][0] - x[i][0]) ** 2 + (x[j][1] - x[i][1]) ** 2)
                if (r <= delta):
                    hor[i].append(j)

        # find neighbor list for old domain
        nei = [[] for _ in range(N1)]
        for i in range(N1):  # loop over all the particle
            for j in range(N1):  # searching neighborhood for each particle
                r = np.sqrt((x[j][0] - x[i][0]) ** 2 + (x[j][1] - x[i][1]) ** 2)
                if (r <= delta and ID1[j]!=2):
                    nei[i].append(j)

        # get all weights for all particles
        all_weights_int = [[] for _ in range(N1)]
        all_weights_dxx = [[] for _ in range(N1)]
        all_weights_dyy = [[] for _ in range(N1)]
        all_weights_dxy = [[] for _ in range(N1)]
        for ind in range(N1):
            if ID1[ind] != 2:
                cur_horizon = hor[ind]
                nh = len(cur_horizon)
                B = np.zeros((basedim, nh))
                for i in range(nh):
                    B[:, i] = self.Phi(x[cur_horizon[i]][0] - x[ind][0], x[cur_horizon[i]][1] - x[ind][1])

                g = self.IPhi_int(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_int[ind] = weights

        for ind in range(N1):
            if ID1[ind] != 2:
                cur_nei = nei[ind]
                nh = len(cur_nei)
                B = np.zeros((basedim, nh))
                for i in range(nh):
                    B[:, i] = self.Phi(x[cur_nei[i]][0] - x[ind][0], x[cur_nei[i]][1] - x[ind][1])

                g = self.IPhi_dxx(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_dxx[ind] = weights

                g = self.IPhi_dyy(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_dyy[ind] = weights

                g = self.IPhi_dxy(0.0, 0.0, delta)
                M = np.matmul(B, B.transpose())
                weights = np.matmul(B.transpose(), np.matmul(np.linalg.pinv(M) + 1e-10 * np.eye(M.shape[0]), g))
                all_weights_dxy[ind] = weights

        sm_diff = np.zeros((N1, N1))
        rhs = np.zeros((N1))
        for i in range(N1):
            if ID1[i] == 0:  # inner  particles
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                rhs[i] = f[i]
                for j in range(nh):
                    sm_diff[i, cur_horizon[j]] += (-2 * 4 / math.pi / delta ** 4) * all_weights_int[i][j]
                    sm_diff[i, i] -= (-2 * 4 / math.pi / delta ** 4) * all_weights_int[i][j]
            elif ID1[i] == 1:  # parts of the horizon is outside the domain
                # if i>52:
                #     print('zaizheintgdun')
                cur_horizon = hor[i]
                nh = len(cur_horizon)
                xn = math.sqrt(x[i][0] ** 2 + x[i][1] ** 2)
                normal_x = 1
                normal_y = 0
                tanvec_x = 0
                tanvec_y = 1
                x_bar = 1.0
                y_bar = x[i][1]
                rhs[i] = f[i]

                # compute M_delta(\xb)
                Md=0.0
                for j in range(nh):
                    if ID1[cur_horizon[j]] == 2:
                        #\int ((y-x)p)^2-((y-x_bar)n)^2+((x-x_bar))^2
                        Md+=(4/math.pi/delta**4)*(((x[cur_horizon[j]][0]-x[i][0])*tanvec_x+(x[cur_horizon[j]][1]-x[i][1])*tanvec_y)**2-((x[cur_horizon[j]][0]-x_bar)*normal_x+(x[cur_horizon[j]][1]-y_bar)*normal_y)**2+((x[i][0]-x_bar)*normal_x+(x[i][1]-y_bar)*normal_y)**2)*all_weights_int[i][j]

                for j in range(nh):
                    g_xbar=((math.cos(x_bar) * math.cos(y_bar))*normal_x+(-math.sin(x_bar) * math.sin(y_bar))*normal_y)
                    g_xbar = (1) * normal_x + (0) * normal_y
                    g_xbar = 0.0
                    if ID1[cur_horizon[j]] != 2:  # for the particles inside the domain (including Neumann bd)
                        # part 1: nonlocal integral
                        sm_diff[i, cur_horizon[j]] += (-2*4/math.pi/delta**4) * all_weights_int[i][j]
                        sm_diff[i, i] -= (-2*4/math.pi/delta**4) * all_weights_int[i][j]
                    else:
                        rhs[i] += (2 * 4/math.pi/delta**4 * ((x[cur_horizon[j]][0]-x[i][0])*normal_x+(x[cur_horizon[j]][1]-x[i][1])*normal_y)) * all_weights_int[i][j] * g_xbar  # du/dn=g
                        rhs[i] -= (4/math.pi/delta**4)*(((x[cur_horizon[j]][0]-x_bar)*normal_x+(x[cur_horizon[j]][1]-y_bar)*normal_y)**2-((x[i][0]-x_bar)*normal_x+(x[i][1]-y_bar)*normal_y)**2)*all_weights_int[i][j] * f[i]
                cur_nei = nei[i]
                nh = len(cur_nei)
                for j in range(nh):
                    # part 2: u(x)_pp
                    sm_diff[i, cur_nei[j]] += -Md * (tanvec_x**2 * all_weights_dxx[i][j] + 2*tanvec_x * tanvec_y * all_weights_dxy[i][j] + tanvec_y ** 2 * all_weights_dyy[i][j])
                    sm_diff[i, i] -= -Md * (tanvec_x**2 * all_weights_dxx[i][j] + 2*tanvec_x * tanvec_y * all_weights_dxy[i][j] + tanvec_y ** 2 * all_weights_dyy[i][j])
            elif ID1[i] == 3:
                sm_diff[i, i] = 1.0
                rhs[i] = math.sin(x[i][0]) * math.cos(x[i][1])
                rhs[i] = x[i][0]
        result = np.matmul(np.linalg.pinv(sm_diff[:, ID1!=2][ID1!=2]), rhs[ID1 != 2])
        uu = np.array(x)
        return result

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size() + (2,) if p.is_complex() else p.size()))
    return c
from queue import PriorityQueue
from generate_yaml import Params
import qimpy as qp
import numpy as np
import matplotlib.pyplot as plt
import torch
from grid1d import Grid1D, get1D
from hard_rods_fmt import HardRodsFMT
from mlcdft import MLCDFT
from nn_function import NNFunction
import h5py

qp.utils.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

def run(*,L,dz,R,T,n_bulk,vsigma,lbda,filename):
    grid1d = Grid1D(L=L, dz=dz)
    cdft = HardRodsFMT(grid1d, R=R, T=T, n_bulk=n_bulk)
    # cdft = MLCDFT(grid1d, T=0.1, n_bulk=0.6, w=NNFunction(1, 2, []), f_ex=NNFunction(2, 2, []))
    qp.log.info(f"mu = {cdft.mu}")

    z1d = get1D(grid1d.z)
    Vsigma = vsigma
    # Vshape = qp.grid.FieldR(grid1d.grid, data=(0.5* ((0.4 * grid1d.L - (grid1d.z - 0.5 * grid1d.L).abs()) / Vsigma).erfc()))
    Vshape = qp.grid.FieldR(grid1d.grid, data=(-0.5 * ((grid1d.z - 0.5 * grid1d.L) / Vsigma).square()).exp())

    n = []
    E = []
    P_lambda = []
    for index in range(lbda['min'],lbda['max'],lbda['min'],lbda['step']): # np.arrange!!!!
        VDimensionless = index/10  #0.1-10.0
        V0 = VDimensionless * cdft.T
        # cdft.V.data = (0.5 * V0) * ((0.4 * grid1d.L - (grid1d.z - 0.5 * grid1d.L).abs()) / Vsigma).erfc()
        cdft.V = V0 * Vshape

        cdft.finite_difference_test(cdft.random_direction())
        E_temp = cdft.minimize()
        n_temp = get1D(cdft.n.data)
        E.append(E_temp)
        n.append(n_temp)
        P_lambda.append(V0)


    f = h5py.File(filename, "w")

    dset = f.create_dataset("z1d", (1,len(z1d)), dtype = 'f')
    dset[...] = z1d

    # f['z1d'] = z1d

    V = get1D(Vshape.data)
    dset2 = f.create_dataset("V", (1,len(V)), dtype = 'f')
    dset2[...] = V

    dset3 = f.create_dataset("lambda", (1,len(P_lambda)), dtype = 'f')
    dset3[...] = P_lambda

    n = np.array(n)
    n_shape = n.shape
    number_of_iter = n_shape[0]
    dset4 = f.create_dataset("n", (number_of_iter,n_shape[1]), dtype = 'f')
    dset4[...] = n
    dset4.attrs["n_bulk"] = 0.6 

    # f['n'] = n
    # f.attrs

    dset5 = f.create_dataset("E", (1,len(E)), dtype = 'f')
    dset5[...] = E

    f.close()
    return

import yaml

with open(r'Params.yaml') as file:
    Params = yaml.load(file, Loader=yaml.FullLoader)

run(**Params)


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

grid1d = Grid1D(L=40., dz=0.01)
cdft = HardRodsFMT(grid1d, R=0.5, T=0.1, n_bulk=0.6)
#cdft = MLCDFT(grid1d, T=0.1, n_bulk=0.6, w=NNFunction(1, 2, []), f_ex=NNFunction(2, 2, []))
qp.log.info(f"mu = {cdft.mu}")

z1d = get1D(grid1d.z)
Vsigma = 0.1
Vshape = qp.grid.FieldR(grid1d.grid, data=(-0.5 * ((grid1d.z - 0.5 * grid1d.L) / Vsigma).square()).exp())

n0 = cdft.n
V0step = 0.5
V0max = 10.
results = {}
for Vsign in (-1, +1):
    cdft.n = n0
    for V0mag in np.arange(0., V0max, V0step):
        V0 = Vsign * V0mag
        cdft.V = V0 * Vshape
        E = cdft.minimize()
        n = cdft.n
        results[V0] = (n, float(E))

P_lambda = np.array(sorted(results.keys()))
n_f = []
E = []
for V0, (n, free_e) in results.items():
    temp_n = get1D(n.data)
    n_f.append(temp_n)
    E.append(free_e)
n_f = np.array(n_f)

f = h5py.File("data.hdf5", "w")

dset = f.create_dataset("z", (1,len(z1d)), dtype = 'f')
dset[...] = z1d

V = get1D(Vshape.data)
dset2 = f.create_dataset("V", (1,len(V)), dtype = 'f')
dset2[...] = V

dset3 = f.create_dataset("lambda", (1,len(P_lambda)), dtype = 'f')
dset3[...] = P_lambda

n_shape = n_f.shape
number_of_iter = n_shape[0]
dset4 = f.create_dataset("n", (number_of_iter,n_shape[1]), dtype = 'f')
dset4[...] = n_f
dset4.attrs["n_bulk"] = 0.6 

dset5 = f.create_dataset("E", (1,len(E)), dtype = 'f')
dset5[...] = E

f.close()

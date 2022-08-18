import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"]=""  # works if used before calling torch which seems to initialize the GPU

import qimpy as qp
import numpy as np
import matplotlib.pyplot as plt
import torch
from grid1d import Grid1D, get1D
from hard_rods_fmt import HardRodsFMT

torch._C._cuda_getDeviceCount()  # 0 now 

# torch._C._cuda_getDeviceCount()  # 1 
# torch.cuda.is_available = lambda : False


qp.utils.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

grid1d = Grid1D(L=40., dz=0.01)
cdft = HardRodsFMT(grid1d, R=0.5, T=0.1, n_bulk=0.6)

# Set external potential:
V0 = 10 * cdft.T
Vsigma = 0.1
cdft.V.data = (0.5 * V0) * (
    (0.4 * grid1d.L - (grid1d.z - 0.5 * grid1d.L).abs()) / Vsigma
).erfc()

# Finite difference test
cdft.finite_difference_test(cdft.random_direction())

cdft.minimize()
n = cdft.n


# Plot density and potential:
z1d = get1D(grid1d.z)
plt.plot(z1d, get1D(cdft.V.data), label="V")
plt.plot(z1d, get1D(n.data), label="n")
# plt.plot(z1d, get1D(n.convolve(w0_tilde).data), label="n0")
# plt.plot(z1d, get1D(n.convolve(w1_tilde).data), label="n1")
plt.axhline(cdft.n_bulk, color='k', ls='dotted')
plt.xlabel("z")
plt.legend()
plt.show()

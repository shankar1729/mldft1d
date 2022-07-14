import qimpy as qp
import numpy as np
import matplotlib.pyplot as plt
import torch
from grid1d import Grid1D, get1D
from hard_rods_fmt import HardRodsFMT


qp.utils.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

grid1d = Grid1D(L=40., dz=0.01)
cdft = HardRodsFMT(grid1d, R=0.5, T=1., n_bulk=0.6)

# Create external potential profile:
Vsigma = 1.0
Vshape = (-0.5 * ((grid1d.z - 0.5*grid1d.L) / Vsigma).square()).exp()
z1d = get1D(grid1d.z)
plt.plot(z1d, get1D(Vshape), color='k', lw=1, ls='dotted')

# Run sequence of calculations with varying strength:
n0 = cdft.n
V0step = 0.5
V0max = 10.
for Vsign in (-1, +1):
    cdft.n = n0
    for V0 in np.arange(0., V0max, V0step):
        cdft.V.data = Vsign * V0 * Vshape
        cdft.minimize()
        n = cdft.n
        plt.plot(z1d, get1D(n.data))

plt.show()

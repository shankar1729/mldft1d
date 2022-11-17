import qimpy as qp
import numpy as np
import matplotlib.pyplot as plt
import torch
from grid1d import Grid1D, get1D
from hard_rods_fmt import HardRodsFMT
from mlcdft import Functional, Minimizer, NNFunction


qp.utils.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

grid1d = Grid1D(L=40., dz=0.01)
# cdft = HardRodsFMT(grid1d, R=0.5, T=0.1, n_bulk=0.6)

cdft = Minimizer(
    functional=Functional(T=0.1, w=NNFunction(1, 2, []), f_ex=NNFunction(2, 2, [])),
    grid1d=grid1d,
    n_bulk=0.6,
)
qp.log.info(f"mu = {cdft.mu}")

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
if qp.rc.is_head:
    z1d = get1D(grid1d.z)
    plt.plot(z1d, get1D(cdft.V.data), label="V")
    plt.plot(z1d, get1D(n.data), label="n")
    # plt.plot(z1d, get1D(n.convolve(w0_tilde).data), label="n0")
    # plt.plot(z1d, get1D(n.convolve(w1_tilde).data), label="n1")
    plt.axhline(cdft.n_bulk, color='k', ls='dotted')
    plt.xlabel("z")
    plt.legend()
    plt.show()

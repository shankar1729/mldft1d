import qimpy as qp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from grid1d import Grid1D, get1D
from hard_rods_fmt import HardRodsFMT


qp.utils.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

grid1d = Grid1D(L=20., dz=0.01)
cdft = HardRodsFMT(grid1d, R=0.5, T=1., n_bulk=0.6)

# Create external potential profile:
Vsigma = 1.0
Vshape = qp.grid.FieldR(
    grid1d.grid, data=(-0.5 * ((grid1d.z - 0.5 * grid1d.L) / Vsigma).square()).exp()
)
z1d = get1D(grid1d.z)

# Run sequence of calculations with varying strength:
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

# Plot density profiles and potential:
plt.figure(1)
# --- initialize colormap to color by V0
V0_all = np.array(sorted(results.keys()))
normalize = mpl.colors.Normalize(vmin=V0_all.min(), vmax=V0_all.max())
cmap = mpl.cm.get_cmap('RdBu')
# --- plot densities
for V0, (n, E) in results.items():
    plt.plot(z1d, get1D(n.data), color=cmap(normalize(V0)), lw=1)
plt.axhline(cdft.n_bulk, color='k', ls='dotted', lw=1)
# --- plot potential for comparison
plt.plot(z1d, get1D(Vshape.data), color='k', lw=1, ls='dashed')
plt.xlabel(r'$z$')
plt.ylabel(r'$n(z)$')
plt.xlim(0, grid1d.L)
plt.ylim(0, None)
# --- add colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=normalize)
sm.set_array([])
plt.colorbar(sm, label=r"Perturbation strength, $V_0$")


def trapz(f: np.ndarray, h: float) -> np.ndarray:
    """Cumulative trapezoidal integral of a function sampled at spacing `h`."""
    return np.concatenate(([0.], np.cumsum(0.5*(f[:-1] + f[1:])) * h))


# Compare exact and thermodynamically-integrated energies:
plt.figure(2)
# --- exact energies
E_all = np.array([E for V0, (n, E) in sorted(results.items())])
E_all -= E_all[np.where(V0_all == 0.)[0]]  # difference from bulk
plt.plot(V0_all, E_all, label='CDFT')
# --- thermodynamic integration
integrand = np.array([Vshape ^ n for V0, (n, E) in sorted(results.items())])
E_TI = trapz(integrand, V0step)
E_TI -= E_TI[np.where(V0_all == 0.)[0]]  # difference from bulk
plt.plot(V0_all, E_TI, 'r+', label='TI')
plt.axhline(0, color='k', lw=1, ls='dotted')
plt.axvline(0, color='k', lw=1, ls='dotted')
plt.legend()
plt.xlim(V0_all.min(), V0_all.max())
plt.xlabel(r'Perturbation strength, $V_0$')
plt.ylabel(r'Free energy change, $\Delta\Phi$')

plt.show()

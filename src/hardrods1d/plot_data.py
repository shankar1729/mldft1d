import h5py
import hardrods1d as hr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_data(data_file: str) -> None:
    f = h5py.File(data_file, "r")
    z = np.array(f["z"])
    V = np.array(f["V"])
    lbda = np.array(f["lbda"])
    n = np.array(f["n"])
    E = np.array(f["E"])
    n_bulk = f.attrs["n_bulk"]

    # Plot density profiles and potential:
    plt.figure(1)
    # --- initialize colormap to color by V0
    normalize = mpl.colors.Normalize(vmin=lbda.min(), vmax=lbda.max())
    cmap = mpl.cm.get_cmap("RdBu")
    # --- plot densities
    for lbda_cur, n_cur in zip(lbda, n):
        plt.plot(z, n_cur, color=cmap(normalize(lbda_cur)), lw=1)
    plt.axhline(n_bulk, color="k", ls="dotted", lw=1)
    # --- plot potential for comparison
    plt.plot(z, V, color="k", lw=1, ls="dashed")
    plt.xlabel(r"$z$")
    plt.ylabel(r"$n(z)$")
    plt.xlim(z.min(), z.max())
    # --- add colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])
    plt.colorbar(sm, label=r"Perturbation strength, $\lambda$")

    # Compare exact and thermodynamically-integrated energies:
    plt.figure(2)
    # --- exact energies
    E -= np.interp(0.0, lbda, E)  # difference from bulk
    plt.plot(lbda, E, label="CDFT")
    # --- thermodynamic integration
    dz = z[1] - z[0]
    integrand = (n @ V) * dz
    dlbda = lbda[1] - lbda[0]
    E_TI = hr.trapz(integrand, dlbda)
    E_TI -= np.interp(0.0, lbda, E_TI)  # difference from bulk
    plt.plot(lbda, E_TI, "r+", label="TI")
    plt.axhline(0, color="k", lw=1, ls="dotted")
    plt.axvline(0, color="k", lw=1, ls="dotted")
    plt.legend()
    plt.xlim(lbda.min(), lbda.max())
    plt.xlabel(r"Perturbation strength, $V_0$")
    plt.ylabel(r"Free energy change, $\Delta\Phi$")

    plt.show()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_data.py <data.h5>")
        exit(1)
    data_file = sys.argv[1]
    plot_data(data_file)


if __name__ == "__main__":
    main()

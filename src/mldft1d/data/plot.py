import sys

import h5py
from matplotlib import colors, cm
import matplotlib.pyplot as plt
import numpy as np


def plot_data(data_file: str) -> None:
    with h5py.File(data_file, "r") as f:
        z = np.array(f["z"])
        V = np.array(f["V"])
        n = np.array(f["n"])
        E = np.array(f["E"])
        dE_dn = np.array(f["dE_dn"])

    n_sites = n.shape[1]
    n_perts = len(E)
    normalize = colors.Normalize(vmin=0, vmax=max(n_perts - 1, 1))
    cmap = cm.get_cmap("RdBu")

    for i_site in range(n_sites):
        # Plot densities
        plt.figure()
        for i_pert, n_cur in enumerate(n):
            plt.plot(z, n_cur[i_site], color=cmap(normalize(i_pert)), lw=1)
        # Plot external potential shape for comparison
        plt.plot(z, V[i_site], color="k", lw=1, ls="dashed")
        plt.xlabel(r"$z$")
        plt.ylabel(r"$n(z)$")
        plt.title(f"Site {i_site} density")
        plt.xlim(z.min(), z.max())
        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=normalize)
        sm.set_array([])
        plt.colorbar(sm, label=r"Calculation index", ax=plt.gca())

        # Plot potentials
        plt.figure()
        for i_pert, dE_dn_cur in enumerate(dE_dn):
            plt.plot(z, dE_dn_cur[i_site], color=cmap(normalize(i_pert)), lw=1)
        # --- plot external potential shape for comparison
        plt.plot(z, V[i_site], color="k", lw=1, ls="dashed")
        plt.xlabel(r"$z$")
        plt.ylabel(r"$\delta E/\delta n(z)$")
        plt.title(f"Site {i_site} potential")
        plt.xlim(z.min(), z.max())
        # --- add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=normalize)
        sm.set_array([])
        plt.colorbar(sm, label=r"Calculation index", ax=plt.gca())

    if n_perts > 1:
        # Check E and dE_dn consistency:
        plt.figure()
        # --- exact energies
        i_perts = np.arange(n_perts)
        plt.plot(i_perts, E - E[0], label="CDFT")
        # --- integration of dE/dn
        dz = z[1] - z[0]
        dn = np.diff(n, axis=0)
        dE_dn_mid = 0.5 * (dE_dn[:-1] + dE_dn[1:])
        integrand = dz * (dn * dE_dn_mid).sum(axis=(1, 2))
        E_TI = np.concatenate(([0.0], np.cumsum(integrand)))
        plt.plot(i_perts, E_TI, "r+", label="TI")
        plt.legend()
        plt.xlim(0, n_perts - 1)
        plt.xlabel(r"Calculation index")
        plt.ylabel(r"Free energy change, $\Delta\Phi$")

    plt.show()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot.py <data.h5>")
        exit(1)
    data_file = sys.argv[1]
    plot_data(data_file)


if __name__ == "__main__":
    main()

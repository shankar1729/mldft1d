import os
import glob

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline

from qimpy.io import Unit

r_unit = Unit.MAP["Angstrom"]
V_unit = Unit.MAP["kcal/mol"]
T_unit = Unit.MAP["K"]
P_unit = 1.01325 * Unit.MAP["bar"]  # atm
n_unit = r_unit ** (-3)

runs_base_path = "Runs"  # Temporary
lbda_arr = np.arange(16) * 0.02  # nominal V amplitude in eV (only used in filenames)

sigma_smooth = 1  # in units of input md grid
down_sample = 2  # reduction in resolution for output

T = 300 * T_unit  # in atomic units

# Bulk EOS properties of water
eos_data = np.loadtxt("Bulk/a_ex.dat").T
n_bulk_data = eos_data[0] * n_unit
a_ex_bulk_data = eos_data[1] * n_unit * V_unit
a_ex_prime_bulk_data = eos_data[2] * V_unit
a_ex_bulk = CubicSpline(n_bulk_data, a_ex_bulk_data)

# Look-up from pressure:
P_data = a_ex_prime_bulk_data * n_bulk_data - a_ex_bulk_data + T * n_bulk_data
i_P_min = np.argmin(P_data)
n_bulk_from_P = CubicSpline(P_data[i_P_min:], n_bulk_data[i_P_min:])


def get_data(prefix: str, lbda: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get grid, ext. potential and density for lbda within prefix."""
    filename = f"{prefix}/data-U{lbda:+.2f}.h5"
    with h5py.File(filename, "r") as fp:
        r = np.array(fp["r"]) * r_unit
        V = np.array(fp["V"])[:, 1] * V_unit  # only keep O
        n = np.array(fp["n"])[:, 1] * n_unit  # only keep O
        n = gaussian_filter1d(n, sigma_smooth)
    return r[::down_sample], V[::down_sample], n[::down_sample]


def get_pressure(src_path: str) -> np.ndarray:
    """Get average pressure for each collection sequence from log file."""
    filename = sorted(glob.glob(f"{src_path}/out.o*"))[-1]  # last log file
    P = []
    collect = False
    for i_line, line in enumerate(open(filename)):
        if line.startswith("Starting collection"):
            collect = True
            P_cur = []
        elif line.startswith("Completed collection"):
            collect = False
            P.append(np.mean(P_cur))
        elif collect and not line.startswith("Cycle"):
            P_cur.append(float(line.split()[2]))
    return np.array(P) * P_unit


def convert(src_path: str, out_file: str) -> None:
    """Convert mdext data from src_path to mldft1d format in out_file."""
    # Get data for all lambdas on common grid:
    V_arr = []
    n_arr = []
    for lbda in lbda_arr:
        r, V, n = get_data(src_path, lbda)
        V_arr.append(V)
        n_arr.append(n)
    V = np.stack(V_arr)
    n = np.stack(n_arr)
    n = np.clip(n, 1e-16, None)  # Floor on zero densities to avoid error in log below

    # Add symmetric image to make even:
    slice_reverse = slice(-2, 0, -1)
    dr = r[1] - r[0]
    L = 2 * r[-1]
    dOmega = dr * (r_unit**2)  # setting transverse size to 1 A (not bohr)
    Omega = L * (r_unit**2)
    r = np.concatenate((r, L - r[slice_reverse]))
    V = np.concatenate((V[:, slice_reverse], V), axis=1)
    n = np.concatenate((n[:, slice_reverse], n), axis=1)

    # Get chemical potential and bulk energy density from pressure
    P = get_pressure(src_path)
    n_bulk = n_bulk_from_P(P)
    mu = a_ex_bulk(n_bulk, nu=1) + T * (np.log(n_bulk / n_unit) + 1.0)  # = da/dn

    # Convert V to excess potential using Euler-Lagrange equation:
    # T(log(n/n0) + 1) + dA_ex/dn + (V_ext - mu) = 0
    V_ex = mu[:, None] - V - T * (1.0 + np.log(n / n_unit))

    # Construct total energies by TI:
    n_mid = 0.5 * (n[:-1] + n[1:])
    delta_V = np.diff(V, axis=0)
    delta_E = (n_mid * delta_V).sum(axis=-1) * dOmega
    n_bulk_0 = np.mean(n[0])
    E0 = Omega * (a_ex_bulk(n_bulk_0) + T * n_bulk_0 * np.log(n_bulk_0 / n_unit))
    E = E0 + np.concatenate(([0], np.cumsum(delta_E)))
    E_ex = E - (n * (T * np.log(n / n_unit) + V)).sum(axis=1) * dOmega

    # Write hdf5 file (back in original units to keep things O(1)):
    with h5py.File(out_file, "w") as fp:
        fp["z"] = r / r_unit
        fp["V"] = delta_V[0, None] / V_unit  # only used in plotting
        fp["n"] = n[:, None] / n_unit
        fp["E"] = E_ex / V_unit
        fp["dE_dn"] = V_ex[:, None] / V_unit
    print(
        f"Wrote {out_file} with L = {L/r_unit:.2f} A and dr = {dr/r_unit:.2f}"
        f" with mu in {mu.min()/V_unit:.3f} to {mu.max()/V_unit:.3f} kcal/mol"
        f" and n_bulk in {n_bulk.min()/n_unit:.4f} to {n_bulk.max()/n_unit:.4f} A^-3."
    )


def main() -> None:
    for seed in range(640):
        out_file = f"Data/random{seed}.h5"
        if os.path.isfile(out_file):
            print(f"Skipped {out_file} (already exists)")
        else:
            try:
                convert(f"{runs_base_path}/seed{seed:04d}", out_file)
            except FileNotFoundError:
                print(f"Failed {out_file} due to insufficient replicas")


if __name__ == "__main__":
    main()

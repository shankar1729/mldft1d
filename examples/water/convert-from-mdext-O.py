import os

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

from qimpy.io import Unit

r_unit = Unit.MAP["Angstrom"]
V_unit = Unit.MAP["kcal/mol"]
T_unit = Unit.MAP["K"]
n_unit = r_unit ** (-3)

mdext_base_path = "/home/kamron/mdext_KF/examples/water/015datasetZ40replicas22Feb"
n_replicas = 4  # number of replicas attempted to be calculated
n_replicas_min = 3  # process data if these many replicas available
lbda_arr = np.arange(11) * 0.02  # nominal V amplitude in eV (only used in filenames)

sigma_smooth = 3  # in units of input md grid
down_sample = 2  # reducistion in resolution for output

n_bulk = 0.0049383  # in atomic units
T = 300 * T_unit  # in atomic units
bulk_ramp = 3.0  # in bohrs

# Bulk EOS properties of water
a_bulk = -0.000049418134  # bulk free energy density (with A_id := T n log n/n_unit)
a_n_bulk = -0.009996667327  # density derivative in bulk condition
mu = a_n_bulk
print(f"mu = {mu:.3f} kcal/mol")


def get_data(prefix: str, lbda: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get grid, ext. potential and replica-averaged density for lbda within prefix."""
    r_arr = []
    V_arr = []
    n_arr = []
    for i_replica in range(n_replicas):
        filename = f"{prefix}/replica{(i_replica + 1):04d}/data-U{lbda:+.2f}.h5"
        if not os.path.isfile(filename):
            continue
        with h5py.File(filename, "r") as fp:
            r = np.array(fp["r"]) * r_unit
            V = np.array(fp["V"])[:, 1] * V_unit  # only keep O
            n = np.array(fp["n"])[:, 1] * n_unit  # only keep O
            x_ramp = (r - r.max()) / bulk_ramp  # dimensionless input to ramp function
            n += (n_bulk - n) * np.exp(-0.5 * x_ramp**2)  # apply bulk ramp
            n = gaussian_filter1d(n, sigma_smooth)
            r_arr.append(r[::down_sample])
            V_arr.append(V[::down_sample])
            n_arr.append(n[::down_sample])

    if len(n_arr) < n_replicas_min:
        raise FileNotFoundError

    # Bring to common grid and average n:
    nr_min = min(len(r) for r in r_arr)
    r = r_arr[0][:nr_min]  # same in all replicas, except padding at far end
    V = V_arr[0][:nr_min]  # same in all replicas, except padding at far end
    n = np.stack([n[:nr_min] for n in n_arr]).mean(axis=0)  # average over replicas
    return r, V, n


def convert(src_path: str, out_file: str) -> None:
    """Convert mdext data from src_path to mldft1d format in out_file."""
    # Get data for all lambdas on common grid:
    r = np.zeros(1)
    V_arr = []
    n_arr = []
    for lbda in lbda_arr:
        r_cur, V, n = get_data(src_path, lbda)
        V_arr.append(V)
        n_arr.append(n)
        if len(r_cur) > len(r):
            r = r_cur  # keep the longest grid

    # Pad V and n to same size across series:
    V = np.zeros((len(lbda_arr), len(r)))
    n = np.full_like(V, n_bulk)
    for i_lbda, (V_cur, n_cur) in enumerate(zip(V_arr, n_arr)):
        len_cur = len(V_cur)
        V[i_lbda, :len_cur] = V_cur
        n[i_lbda, :len_cur] = n_cur

    # Add symmetric image to make even:
    slice_reverse = slice(-2, 0, -1)
    dr = r[1] - r[0]
    L = 2 * r[-1]
    dOmega = dr * (r_unit**2)  # setting transverse size to 1 A (not bohr)
    Omega = L * (r_unit**2)
    r = np.concatenate((r, L - r[slice_reverse]))
    V = np.concatenate((V[:, slice_reverse], V), axis=1)
    n = np.concatenate((n[:, slice_reverse], n), axis=1)

    # Convert V to excess potential using Euler-Lagrange equation:
    # T(log(n) + 1) + dA_ex/dn + (V_ext - mu) = 0
    V_ex = mu - V - T * (1.0 + np.log(n))

    # Construct total energies by TI:
    n_mid = 0.5 * (n[:-1] + n[1:])
    delta_V = np.diff(V, axis=0)
    delta_E = (n_mid * delta_V).sum(axis=-1) * dOmega
    E0 = Omega * (a_bulk - mu * n_bulk)
    E = E0 + np.concatenate(([0], np.cumsum(delta_E)))
    E_ex = E - (n * (T * np.log(n) + V - mu)).sum(axis=1) * dOmega

    # Write hdf5 file (back in original units to keep things O(1)):
    with h5py.File(out_file, "w") as fp:
        fp["z"] = r / r_unit
        fp["V"] = delta_V[0, None] / V_unit  # only used in plotting
        fp["n"] = n[:, None] / n_unit
        fp["E"] = E_ex / V_unit
        fp["dE_dn"] = V_ex[:, None] / V_unit
    print(
        f"Wrote {out_file} with grid length {L/r_unit:.2f} A"
        f" and spacing {dr/r_unit:.2f} A."
    )


def main() -> None:
    for seed in range(288):
        out_file = f"random-spce-O/random{seed}.h5"
        if os.path.isfile(out_file):
            print(f"Skipped {out_file} (already exists)")
        else:
            try:
                convert(f"{mdext_base_path}/seed{seed:04d}", out_file)
            except FileNotFoundError:
                print(f"Failed {out_file} due to insufficient replicas")


if __name__ == "__main__":
    main()

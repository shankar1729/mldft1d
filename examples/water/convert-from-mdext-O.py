import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from qimpy.io import Unit

r_unit = Unit.MAP["Angstrom"]
V_unit = Unit.MAP["kcal/mol"]
n_unit = r_unit ** (-3)

mdext_base_path = "/home/kamron/mdext_KF/examples/water/015datasetZ40replicas22Feb"
n_replicas = 4
lbda_arr = np.arange(11) * 0.02  # nominal V amplitude in eV (only used in filenames)

sigma_smooth = 3  # in units of input md grid
down_sample = 2  # reducistion in resolution for output

n_bulk = 0.0049383  # in atomic units
bulk_ramp = 3.0  # in bohrs


def get_data(prefix: str, lbda: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get grid, ext. potential and replica-averaged density for lbda within prefix."""
    r_arr = []
    V_arr = []
    n_arr = []
    for i_replica in range(n_replicas):
        filename = f"{prefix}/replica{(i_replica + 1):04d}/data-U{lbda:+.2f}.h5"
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
    r = np.concatenate((r, L - r[slice_reverse]))
    V = np.concatenate((V, V[:, slice_reverse]), axis=1)
    n = np.concatenate((n, n[:, slice_reverse]), axis=1)
    print(f"Grid with length {L/r_unit:.2f} A and spacing {dr/r_unit:.2f} A.")

    plt.figure()
    plt.plot(r, n.T)
    plt.ylim(0, None)
    plt.axhline(n_bulk, color="k", ls="dotted")
    plt.savefig("test_n.pdf", bbox_inches="tight")

    plt.figure()
    plt.plot(r, V.T)
    plt.savefig("test_V.pdf", bbox_inches="tight")


def main() -> None:
    convert(f"{mdext_base_path}/seed0007", None)


if __name__ == "__main__":
    main()

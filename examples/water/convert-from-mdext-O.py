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

sigma_smooth = 3  # in units of input md grid
down_sample = 2  # reducistion in resolution for output

n_bulk = 0.0049383  # in atomic units
bulk_ramp = 3.0  # in bohrs


def get_data(prefix: str, lbda: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract density and excess potential averaged within prefix for strength lbda."""
    r_arr = []
    V_arr = []
    n_arr = []
    for i_replica in range(n_replicas):
        filename = f"{prefix}/replica{(i_replica + 1):04d}/data-U{lbda:+.2f}.h5"
        with h5py.File(filename, "r") as fp:
            r_arr.append(np.array(fp["r"])[::down_sample] * r_unit)
            V_arr.append(np.array(fp["V"])[::down_sample, 1] * V_unit)  # only keep O
            n_arr.append(
                gaussian_filter1d(np.array(fp["n"])[:, 1], sigma_smooth)[::down_sample]
                * n_unit
            )

    # Bring to common grid and average n:
    nr_min = min(len(r) for r in r_arr)
    r = r_arr[0][:nr_min]  # same in all replicas, except padding at far end
    V = V_arr[0][:nr_min]  # same in all replicas, except padding at far end
    n = np.stack([n[:nr_min] for n in n_arr]).mean(axis=0)  # average over replicas
    n_err = np.stack([n[:nr_min] for n in n_arr]).std(axis=0) / np.sqrt(n_replicas)

    # Apply bulk ramp:
    weight_bulk = np.exp((r - r.max()) / bulk_ramp)
    n += weight_bulk * (n_bulk - n)

    plt.figure()
    plt.fill_between(r, n - n_err, n + n_err, color="r", alpha=0.3)
    plt.plot(r, n, "r")
    plt.ylim(0, None)
    plt.axhline(n_bulk, color="k", ls="dotted")
    plt.savefig("test_n.pdf", bbox_inches="tight")

    plt.figure()
    plt.plot(r, V)
    plt.savefig("test_V.pdf", bbox_inches="tight")
    # *


def convert(src_path: str, out_file: str) -> None:
    """Convert mdext data from src_path to mldft1d format in out_file."""


def main() -> None:
    get_data(f"{mdext_base_path}/seed0191", 0.20)


if __name__ == "__main__":
    main()

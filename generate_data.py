import qimpy as qp
import numpy as np
import torch
from grid1d import Grid1D, get1D
from hard_rods_fmt import HardRodsFMT
from typing import Optional
import h5py
import yaml
import sys


def run(
    *,
    L: float,
    dz: float,
    R: float,  #: Radius / half-length `R`
    T: float,  #: Temperature
    n_bulk: float,  #: Bulk number density of the fluid
    Vshape: dict,
    lbda: dict,  #: min: float, max: float, step: float,
    filename: str,  #: hdf5 filename, must end with .hdf5
) -> None:

    grid1d = Grid1D(L=L, dz=dz)
    cdft = HardRodsFMT(grid1d, R=R, T=T, n_bulk=n_bulk)
    n0 = cdft.n

    qp.log.info(f"mu = {cdft.mu}")
    V = qp.grid.FieldR(grid1d.grid, data=get_V(grid1d.z, grid1d.L, **Vshape))

    lbda_arr = np.arange(lbda["min"], lbda["max"], lbda["step"])
    E = np.zeros_like(lbda_arr)
    n = np.zeros((len(E), len(get1D(grid1d.z))))

    # Split runs by sign and in increasing order of perturbation strength:
    abs_index = abs(lbda_arr).argsort()
    pos_index = [i for i in abs_index if lbda_arr[i] >= 0.0]
    neg_index = [i for i in abs_index if lbda_arr[i] < 0.0]
    for cur_index in (neg_index, pos_index):
        cdft.n = n0
        for index in cur_index:
            cdft.V = lbda_arr[index] * V
            E[index] = float(cdft.minimize())
            n[index] = get1D(cdft.n.data)

    f = h5py.File(filename, "w")
    f["z"] = get1D(grid1d.z)
    f["V"] = get1D(V.data)
    f["lbda"] = lbda_arr
    f["n"] = n
    f["E"] = E
    f.attrs["n_bulk"] = n_bulk
    f.attrs["T"] = T
    f.attrs["R"] = R
    f.close()


def get_V(
    z: torch.Tensor,
    L: float,
    *,
    gauss: Optional[dict] = None,
    cosine: Optional[dict] = None,
) -> None:
    # Make sure exactly one is specified
    assert len([x for x in (gauss, cosine) if (x is not None)]) == 1

    if gauss is not None:
        return (-0.5 * ((z - 0.5 * L) / gauss["sigma"]).square()).exp()

    if cosine is not None:
        pass  # TODO


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python generate_data.py <input.yaml>")
        exit(1)
    in_file = sys.argv[1]

    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    with open(in_file) as fp:
        run(**yaml.load(fp, Loader=yaml.FullLoader))


if __name__ == "__main__":
    main()

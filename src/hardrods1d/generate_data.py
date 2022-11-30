import qimpy as qp
import numpy as np
import hardrods1d as hr
import torch
import h5py
import sys
from typing import Dict, Any


def run(
    *,
    L: float,
    dz: float,
    R: float,  #: Radius / half-length `R`
    T: float,  #: Temperature
    n_bulk: float,  #: Bulk number density of the fluid
    Vshape: dict,  #: Potential shape and parameters
    lbda: dict,  #: min: float, max: float, step: float,
    filename: str,  #: hdf5 filename, must end with .hdf5
) -> None:

    grid1d = hr.Grid1D(L=L, dz=dz)
    cdft = hr.HardRodsFMT(grid1d, R=R, T=T, n_bulk=n_bulk)
    n0 = cdft.n

    qp.log.info(f"mu = {cdft.mu}")
    V = qp.grid.FieldR(grid1d.grid, data=get_V(grid1d.z, grid1d.L, **Vshape))

    lbda_arr = np.arange(lbda["min"], lbda["max"], lbda["step"])
    E = np.zeros_like(lbda_arr)
    n = np.zeros((len(E), len(hr.get1D(grid1d.z))))

    # Split runs by sign and in increasing order of perturbation strength:
    abs_index = abs(lbda_arr).argsort()
    pos_index = [i for i in abs_index if lbda_arr[i] >= 0.0]
    neg_index = [i for i in abs_index if lbda_arr[i] < 0.0]
    for cur_index in (neg_index, pos_index):
        cdft.n = n0
        for index in cur_index:
            cdft.V = lbda_arr[index] * V
            E[index] = float(cdft.minimize())
            n[index] = hr.get1D(cdft.n.data)

    f = h5py.File(filename, "w")
    f["z"] = hr.get1D(grid1d.z)
    f["V"] = hr.get1D(V.data)
    f["lbda"] = lbda_arr
    f["n"] = n
    f["E"] = E
    f.attrs["n_bulk"] = n_bulk
    f.attrs["T"] = T
    f.attrs["R"] = R
    f.close()


def get_V(z: torch.Tensor, L: float, *, shape: str, **kwargs) -> torch.Tensor:
    return get_V_map[shape](z, L, **kwargs)  # type: ignore


def get_V_gauss(z: torch.Tensor, L: float, *, sigma: float) -> torch.Tensor:
    return (-0.5 * ((z - 0.5 * L) / sigma).square()).exp()


def get_V_cosine(z: torch.Tensor, L: float, *, n: int = 1) -> torch.Tensor:
    return ((2 * n * np.pi / L) * z).cos()


get_V_map: Dict[str, Any] = {"gauss": get_V_gauss, "cosine": get_V_cosine}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python generate_data.py <input.yaml>")
        exit(1)
    in_file = sys.argv[1]

    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    input_dict = qp.utils.dict.key_cleanup(qp.utils.yaml.load(in_file))
    run(**input_dict)


if __name__ == "__main__":
    main()

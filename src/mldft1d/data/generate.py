from __future__ import annotations
from typing import Callable, Protocol, Optional, Any, Union, Sequence
from dataclasses import dataclass
import sys

import numpy as np
import torch
import h5py

import qimpy
from qimpy import log, rc, io
from qimpy.mpi import TaskDivision
from qimpy.grid import FieldR
from .. import Grid1D, get1D
from . import v_shape
from .. import hardrods, kohnsham, ising, hf, protocols


make_exact_dft_map: dict[str, Callable[..., protocols.DFT]] = {
    "hardrods": hardrods.make_dft.exact,
    "kohnsham": kohnsham.make_dft.exact,
    "ising": ising.make_dft.exact,
    "hf": hf.make_dft.exact,
}  #: Recognized exact DFTs that can be used for data generation


def run(
    *,
    L: float,
    dz: float,
    n_bulk: Union[float, Sequence[float]],  #: Bulk number density of the fluid
    Vshape: Union[dict, Sequence[dict]],  #: Potential shape and parameters
    lbda: dict,  #: min: float, max: float, step: float,
    filename: str,  #: hdf5 filename, must end with .hdf5
    functional: str,  #: name of exact functional to use (hardrods | kohnsham)
    dn_electrons_dlbda: float = 0.0,  #: optional change of electron count (for hf only)
    **dft_kwargs,  #: extra keyword arguments forwarded to the exact dft
) -> None:

    # Check site density / potential counts:
    Vshape = [Vshape] if isinstance(Vshape, dict) else Vshape
    n_bulks = torch.tensor(
        [n_bulk] if isinstance(n_bulk, float) else n_bulk, device=rc.device
    )
    n_sites = len(n_bulks)
    assert len(Vshape) == n_sites

    grid1d = Grid1D(L=L, dz=dz, parallel=False)
    dft = make_exact_dft_map[functional](grid1d=grid1d, n_bulk=n_bulks, **dft_kwargs)
    n0 = dft.n
    n_electrons0 = n_bulks[0].item() * L

    # Initialize potential shape:
    Vdata = torch.stack(
        [
            v_shape.get(grid1d, **io.dict.key_cleanup(Vshape_i)).data
            for Vshape_i in Vshape
        ]
    )
    V = FieldR(grid1d.grid, data=Vdata)

    lbda_arr = get_lbda_arr(**io.dict.key_cleanup(lbda))
    E = np.zeros_like(lbda_arr)  # part to be trained
    n = np.zeros((len(E), n_sites, len(get1D(grid1d.z))))
    dE_dn = np.zeros_like(n)  # functional derivative

    # Split runs by sign and in increasing order of perturbation strength:
    abs_index = abs(lbda_arr).argsort()
    pos_index = [i for i in abs_index if lbda_arr[i] >= 0.0]
    neg_index = [i for i in abs_index if lbda_arr[i] < 0.0]
    for cur_index in (neg_index, pos_index):
        dft.n = n0
        for index in cur_index:
            lbda_cur = lbda_arr[index]
            dft.V = lbda_cur * V
            if dn_electrons_dlbda and isinstance(dft, hf.DFT):
                dft.n_electrons = n_electrons0 + dn_electrons_dlbda * lbda_cur
            dft.minimize()
            n[index] = get1D(dft.n.data)
            E[index], dE_dn_cur = dft.training_targets()
            dE_dn[index] = get1D(dE_dn_cur.data)

    f = h5py.File(filename, "w")
    f["z"] = get1D(grid1d.z)
    f["V"] = get1D(
        Vdata + (dft.Vnuc.data if isinstance(dft, hf.DFT) else torch.zeros_like(Vdata))
    )
    f["n"] = n
    f["E"] = E
    f["dE_dn"] = dE_dn
    for dft_arg_name, dft_arg_value in dft_kwargs.items():
        if not (isinstance(dft_arg_value, dict) or isinstance(dft_arg_value, list)):
            f.attrs[dft_arg_name] = dft_arg_value
    f.close()


def get_lbda_arr(*, min: float, max: float, step: float) -> np.ndarray:
    n_lbda = int(np.round((max - min) / step))
    return min + step * np.arange(n_lbda + 1)


class Sampler(Protocol):
    """Random sampler for batched-data generation.
    Compatible with scipy.stats distributions."""

    def rvs(self) -> float:
        ...


@dataclass
class Choice:
    """Random choice from a sequence of variables."""

    choices: Sequence
    probabilities: Optional[np.ndarray] = None

    def rvs(self) -> Any:
        index = np.random.choice(len(self.choices), p=self.probabilities, replace=True)
        return self.choices[index]


def batch(n_batch: int, prefix: str, **kwargs) -> None:
    """Batched data generation.
    Call `run` n_batch times and write output to `prefix``i_batch`.h5.
    Every argument with `kwargs` that is an object with an `rvs` method
    will be sampled, while everything else will be forwarded to `run`
    """
    comm = rc.comm
    division = TaskDivision(n_tot=n_batch, n_procs=comm.size, i_proc=comm.rank)
    for i_batch in range(1 + division.i_start, 1 + division.i_stop):
        log.warning(
            f"\n---- Generating {i_batch} of {n_batch} on process {comm.rank} ----\n"
        )
        sampled_args = sample_dict(kwargs)
        try:
            run(filename=f"{prefix}{i_batch}.h5", **sampled_args)
        except Exception as err:
            log.error(f"Failed with exception {err}")


def sample_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively sample Sampler objects within a dictionary."""
    output: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            output[key] = sample_dict(value)
        elif isinstance(value, list):
            output[key] = sample_list(value)
        elif hasattr(value, "rvs"):
            output[key] = value.rvs()
        else:
            output[key] = value
    return output


def sample_list(d: list) -> list:
    """Recursively sample Sampler objects within a list"""
    output: list[Any] = []
    for value in d:
        if isinstance(value, dict):
            output.append(sample_dict(value))
        elif isinstance(value, list):
            output.append(sample_list(value))
        elif hasattr(value, "rvs"):
            output.append(value.rvs())
        else:
            output.append(value)
    return output


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python generate.py <input.yaml>")
        exit(1)
    in_file = sys.argv[1]

    io.log_config()  # default set up to log from MPI head alone
    log.info("Using QimPy " + qimpy.__version__)
    rc.init()

    input_dict = io.dict.key_cleanup(io.yaml.load(in_file))
    run(**input_dict)


if __name__ == "__main__":
    main()

import qimpy as qp
import numpy as np
from .. import Grid1D, get1D
from . import v_shape
from .. import hardrods, kohnsham, ising, protocols
from typing import Callable, Protocol, Optional, Any
from dataclasses import dataclass
import h5py
import sys


make_exact_dft_map: dict[str, Callable[..., protocols.DFT]] = {
    "hardrods": hardrods.make_dft.exact,
    "kohnsham": kohnsham.make_dft.exact,
    "ising": ising.make_dft.exact,
}  #: Recognized exact DFTs that can be used for data generation


def run(
    *,
    L: float,
    dz: float,
    n_bulk: float,  #: Bulk number density of the fluid
    Vshape: dict,  #: Potential shape and parameters
    lbda: dict,  #: min: float, max: float, step: float,
    filename: str,  #: hdf5 filename, must end with .hdf5
    functional: str,  #: name of exact functional to use (hardrods | kohnsham)
    **dft_kwargs,  #: extra keyword arguments forwarded to the exact dft
) -> None:

    grid1d = Grid1D(L=L, dz=dz, parallel=False)
    dft = make_exact_dft_map[functional](grid1d=grid1d, n_bulk=n_bulk, **dft_kwargs)
    n0 = dft.n

    qp.log.info(f"mu = {dft.mu}")
    V = v_shape.get(grid1d, **qp.utils.dict.key_cleanup(Vshape))

    lbda_arr = get_lbda_arr(**qp.utils.dict.key_cleanup(lbda))
    E = np.zeros_like(lbda_arr)
    n = np.zeros((len(E), len(get1D(grid1d.z))))
    E0 = np.zeros_like(E)
    V0 = np.zeros_like(n)
    has_known_part = False

    # Split runs by sign and in increasing order of perturbation strength:
    abs_index = abs(lbda_arr).argsort()
    pos_index = [i for i in abs_index if lbda_arr[i] >= 0.0]
    neg_index = [i for i in abs_index if lbda_arr[i] < 0.0]
    for cur_index in (neg_index, pos_index):
        dft.n = n0
        for index in cur_index:
            dft.V = lbda_arr[index] * V
            E[index] = float(dft.minimize())
            n[index] = get1D(dft.n.data)
            if (EV0_i := dft.known_part()) is not None:
                E0_i, V0_i = EV0_i
                E0[index] = E0_i
                V0[index] = get1D(V0_i.data)
                has_known_part = True

    f = h5py.File(filename, "w")
    f["z"] = get1D(grid1d.z)
    f["V"] = get1D(V.data)
    f["lbda"] = lbda_arr
    f["n"] = n
    f["E"] = E
    if has_known_part:
        f["E0"] = E0
        f["V0"] = V0
    f.attrs["mu"] = dft.mu
    f.attrs["n_bulk"] = n_bulk
    for dft_arg_name, dft_arg_value in dft_kwargs.items():
        f.attrs[dft_arg_name] = dft_arg_value
    f.close()


def get_lbda_arr(*, min: float, max: float, step: float) -> np.ndarray:
    n_lbda = int(np.round((max - min) / step))
    return min + step * np.arange(n_lbda + 1)


class Sampler(Protocol):
    """Random sampler for batched-data generation.
    Compatibly with scipy.stats distributions."""

    def rvs(self) -> float:
        ...


@dataclass
class Choice:
    """Random choice from a sequence of variables."""

    choices: np.ndarray
    probabilities: Optional[np.ndarray] = None

    def rvs(self) -> float:
        return np.random.choice(self.choices, p=self.probabilities, replace=True)


def batch(n_batch: int, prefix: str, **kwargs) -> None:
    """Batched data generation.
    Call `run` n_batch times and write output to `prefix``i_batch`.h5.
    Every argument with `kwargs` that is an object with an `rvs` method
    will be sampled, while everything else will be forwarded to `run`
    """
    comm = qp.rc.comm
    division = qp.utils.TaskDivision(n_tot=n_batch, n_procs=comm.size, i_proc=comm.rank)
    for i_batch in range(1 + division.i_start, 1 + division.i_stop):
        qp.log.warning(
            f"\n---- Generating {i_batch} of {n_batch} on process {comm.rank} ----\n"
        )
        sampled_args = sample_dict(kwargs)
        try:
            run(filename=f"{prefix}{i_batch}.h5", **sampled_args)
        except Exception as err:
            qp.log.error(f"Failed with exception {err}")


def sample_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively sample Sampler objects within a dictionary."""
    output: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            output[key] = sample_dict(value)
        elif hasattr(value, "rvs"):
            output[key] = value.rvs()
        else:
            output[key] = value
    return output


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python generate.py <input.yaml>")
        exit(1)
    in_file = sys.argv[1]

    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    input_dict = qp.utils.dict.key_cleanup(qp.utils.yaml.load(in_file))
    run(**input_dict)


if __name__ == "__main__":
    main()

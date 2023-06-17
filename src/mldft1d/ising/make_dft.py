from .. import Grid1D, Minimizer, protocols, nn
from . import Exact
import qimpy as qp
import torch


def exact(*, grid1d: Grid1D, n_bulk: float, T: float, J: float) -> protocols.DFT:
    """Make exact Ising solver."""
    return Exact(grid1d, n_bulk=n_bulk, T=T, J=J)


def ml(
    *, grid1d: Grid1D, n_bulk: float, load_file: str, T: float, **kwargs
) -> protocols.DFT:
    """Make approximate Ising solver using an ML functional."""
    return Minimizer(
        functionals=[
            nn.Functional.load(
                qp.rc.comm, load_file=load_file, attr_names=("T",), attrs=(T,)
            )
        ],
        grid1d=grid1d,
        n_bulk=n_bulk,
        name="MLIDFT",
        state_to_n=torch.special.expit,
        n_to_state=torch.special.logit,
        **kwargs,
    )

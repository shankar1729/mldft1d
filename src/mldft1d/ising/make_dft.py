from .. import Grid1D, Minimizer, protocols, nn
from . import Exact
import qimpy as qp


def exact(*, grid1d: Grid1D, n_bulk: float, T: float, J: float) -> protocols.DFT:
    """Make exact Ising solver."""
    return Exact(grid1d, n_bulk=n_bulk, T=T, J=J)


def ml(
    *, grid1d: Grid1D, n_bulk: float, load_file: str, T: float, **kwargs
) -> protocols.DFT:
    """Make approximate Kohn-Sham DFT using an ML kinetic energy functional."""
    return Minimizer(
        functionals=[
            nn.Functional.load(
                qp.rc.comm, load_file=load_file, attr_names=("T",), attrs=(T,)
            )
        ],
        grid1d=grid1d,
        n_bulk=n_bulk,
        name="MLIDFT",
        **kwargs,
    )

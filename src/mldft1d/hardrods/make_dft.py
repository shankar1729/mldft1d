from .. import Grid1D, Minimizer, protocols, nn
from . import IdealGas, FMT
import qimpy as qp


def exact(
    *, grid1d: Grid1D, n_bulk: float, T: float, R: float, **kwargs
) -> protocols.DFT:
    """Make hard-rods exact DFT."""
    return Minimizer(
        functionals=(IdealGas(T), FMT(grid1d, T=T, R=R)),
        grid1d=grid1d,
        n_bulk=n_bulk,
        name="FMT",
        **kwargs,
    )


def ml(
    *, grid1d: Grid1D, n_bulk: float, T: float, load_file: str, **kwargs
) -> protocols.DFT:
    """Make hard-rods approximate DFT using an ML excess functional."""
    return Minimizer(
        functionals=(IdealGas(T), nn.Functional.load(qp.rc.comm, load_file=load_file)),
        grid1d=grid1d,
        n_bulk=n_bulk,
        name="MLCDFT",
        **kwargs,
    )

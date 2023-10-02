from .. import Grid1D, protocols, nn
from . import DFT

from qimpy import rc


def exact(
    *, grid1d: Grid1D, n_bulk: float, T: float, n_electrons: float = 0.0
) -> protocols.DFT:
    """Make exact (EXX-OEP) Hartree-Fock solver."""
    return DFT(
        grid1d, exchange_functional="exact", n_bulk=n_bulk, n_electrons=n_electrons, T=T
    )


def lda(
    *, grid1d: Grid1D, n_bulk: float, T: float, n_electrons: float = 0.0
) -> protocols.DFT:
    """Make LDA approximation to Hartree-Fock."""
    return DFT(
        grid1d, exchange_functional="lda", n_bulk=n_bulk, n_electrons=n_electrons, T=T
    )


def ml(
    *,
    grid1d: Grid1D,
    n_bulk: float,
    T: float,
    load_file: str,
    n_electrons: float = 0.0,
    **kwargs,
) -> protocols.DFT:
    """Make ML approximation to Hartree-Fock."""
    return DFT(
        grid1d,
        exchange_functional=nn.Functional.load(rc.comm, load_file=load_file),
        n_electrons=n_electrons,
        n_bulk=n_bulk,
        T=T,
    )

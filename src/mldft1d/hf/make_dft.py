from .. import Grid1D, protocols
from . import DFT


def lda(*, grid1d: Grid1D, n_bulk: float, T: float) -> protocols.DFT:
    """Make LDA approximation to Hartree-Fock."""
    return DFT(grid1d, n_bulk=n_bulk, T=T)

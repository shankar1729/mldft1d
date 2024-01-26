from __future__ import annotations

import torch

from qimpy import rc
from .. import Grid1D, protocols, nn
from . import DFT


def exact(*, grid1d: Grid1D, n_bulk: torch.Tensor, T: float, **kwargs) -> protocols.DFT:
    """Make exact (EXX-OEP) Hartree-Fock solver."""
    return DFT(grid1d, exchange_functional="exact", n_bulk=n_bulk, T=T, **kwargs)


def lda(*, grid1d: Grid1D, n_bulk: torch.Tensor, T: float, **kwargs) -> protocols.DFT:
    """Make LDA approximation to Hartree-Fock."""
    return DFT(grid1d, exchange_functional="lda", n_bulk=n_bulk, T=T, **kwargs)


def ml(
    *, grid1d: Grid1D, n_bulk: torch.Tensor, T: float, load_file: str, **kwargs
) -> protocols.DFT:
    """Make ML approximation to Hartree-Fock."""
    return DFT(
        grid1d,
        exchange_functional=nn.Functional.load(rc.comm, load_file=load_file),
        n_bulk=n_bulk,
        T=T,
        **kwargs,
    )

from __future__ import annotations

import torch

from qimpy import rc
from .. import Grid1D, Minimizer, protocols, nn
from . import Schrodinger, ThomasFermi


def exact(*, grid1d: Grid1D, n_bulk: torch.Tensor, T: float, **kwargs) -> protocols.DFT:
    """Make exact Kohn-Sham DFT (Schrodinger solver)."""
    return Schrodinger(grid1d, n_bulk=n_bulk, T=T, **kwargs)


def lda(*, grid1d: Grid1D, n_bulk: torch.Tensor, T: float, **kwargs) -> protocols.DFT:
    """Make LDA (Thomas-Fermi) density-only DFT approximation."""
    return Minimizer(
        functionals=[ThomasFermi(T)], grid1d=grid1d, n_bulk=n_bulk, name="LDA", **kwargs
    )


def ml(
    *, grid1d: Grid1D, n_bulk: torch.Tensor, load_file: str, **kwargs
) -> protocols.DFT:
    """Make approximate Kohn-Sham DFT using an ML kinetic energy functional."""
    return Minimizer(
        functionals=[nn.Functional.load(rc.comm, load_file=load_file)],
        grid1d=grid1d,
        n_bulk=n_bulk,
        name="MLEDFT",
        **kwargs,
    )

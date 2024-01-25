from __future__ import annotations

import torch

from qimpy import rc
from .. import Grid1D, Minimizer, protocols, nn
from . import IdealGas, FMT


def exact(
    *, grid1d: Grid1D, n_bulk: torch.Tensor, T: float, R: float, **kwargs
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
    *, grid1d: Grid1D, n_bulk: torch.Tensor, T: float, load_file: str, **kwargs
) -> protocols.DFT:
    """Make hard-rods approximate DFT using an ML excess functional."""
    return Minimizer(
        functionals=(
            IdealGas(T),
            nn.Functional.load(rc.comm, load_file=load_file, n_sites=len(n_bulk)),
        ),
        grid1d=grid1d,
        n_bulk=n_bulk,
        name="MLCDFT",
        **kwargs,
    )

from __future__ import annotations
from typing import Union, Sequence
from dataclasses import dataclass

import torch

from qimpy import rc
from qimpy.grid import FieldR
from .. import Grid1D


class FMT:
    """
    Exact 1D hard rods excess functional. [Percus 1969]
    Calling it FMT, because it is the Fundamental Measure Theory in 1D.
    """

    grid1d: Grid1D
    f_bulk: BulkHardRods
    R: torch.Tensor  #: Hard rod radii / half-lengths (one per species for mixture)
    w0_tilde: torch.Tensor  #: 0-dim weight functions (surface measure of FMT in 1D)
    w1_tilde: torch.Tensor  #: 1-dim weight functions (volume measure of FMT in 1D)

    def __init__(
        self, grid1d: Grid1D, *, R: Union[float, Sequence[float]], T: float
    ) -> None:
        """Initializes to bulk fluid with no external potential."""
        self.grid1d = grid1d
        self.R = torch.tensor([R] if isinstance(R, float) else R, device=rc.device)
        self.f_bulk = BulkHardRods(R=self.R, T=T)

        # Initialize FMT weight functions:
        R_bcast = self.R.view(-1, 1, 1, 1)  # add dims to broadcast with Gmag
        GR = grid1d.Gmag * R_bcast
        self.w0_tilde = 2 * GR.cos()  # FT of w0(x) = delta(R-x)
        self.w1_tilde = 2 * R_bcast * GR.sinc()  # FT of w1(x) = theta(R-x)

    def get_energy(self, n: FieldR) -> torch.Tensor:
        T = self.f_bulk.T
        n0 = sum_sites(n.convolve(self.w0_tilde))
        n1 = sum_sites(n.convolve(self.w1_tilde))
        return (-0.5 * T) * (n0 ^ (1.0 - n1).log())

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        return self.f_bulk.get_energy_bulk(n)


@dataclass
class BulkHardRods:
    """Exact bulk free energy of hard rods fluid in 1D."""

    R: torch.Tensor  #: Radius / half-lengths `R` for each species in mixture
    T: float  #: Temperature

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        n0 = 2 * n.sum(dim=-1)
        n1 = 2 * (n @ self.R)
        return -0.5 * self.T * n0 * (1.0 - n1).log()

    def get_energy(self, n: torch.Tensor) -> torch.Tensor:
        n0 = 2 * n.data.sum(dim=0)
        n1 = 2 * (self.R @ n.data)
        return -0.5 * self.T * FieldR(n.grid, data=(n0 * (1.0 - n1).log())).integral()


def sum_sites(n: FieldR) -> FieldR:
    return FieldR(n.grid, data=n.data.sum(dim=-4))

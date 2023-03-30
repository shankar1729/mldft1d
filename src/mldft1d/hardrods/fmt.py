from __future__ import annotations
import qimpy as qp
import torch
from mldft1d import Grid1D
from dataclasses import dataclass


class FMT:
    """
    Exact 1D hard rods excess functional. [Percus 1969]
    Calling it FMT, because it is the Fundamental Measure Theory in 1D.
    """

    grid1d: Grid1D
    f_bulk: BulkHardRods
    T: float  #: Temperature
    w0_tilde: torch.Tensor  #: 0-dim weight function (surface measure of FMT in 1D)
    w1_tilde: torch.Tensor  #: 1-dim weight function (volume measure of FMT in 1D)

    def __init__(self, grid1d: Grid1D, *, R: float, T: float) -> None:
        """Initializes to bulk fluid with no external potential."""
        self.grid1d = grid1d
        self.f_bulk = BulkHardRods(R=R, T=T)

        # Initialize FMT weight functions:
        self.w0_tilde = 2 * (grid1d.Gmag * R).cos()  # FT of w0(x) = delta(R-x)
        self.w1_tilde = 2 * R * (grid1d.Gmag * R).sinc()  # FT of w1(x) = theta(R-x)

    def get_energy(self, n: qp.grid.FieldR) -> torch.Tensor:
        T = self.f_bulk.T
        n0 = n.convolve(self.w0_tilde)
        n1 = n.convolve(self.w1_tilde)
        return (-0.5 * T) * (n0 ^ (1.0 - n1).log())

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        return self.f_bulk.get_energy_bulk(n)


@dataclass
class BulkHardRods:
    """Exact bulk free energy of hard rods fluid in 1D."""

    R: float  #: Radius / half-length `R`
    T: float  #: Temperature

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        return -self.T * n * (1.0 - 2 * self.R * n).log()

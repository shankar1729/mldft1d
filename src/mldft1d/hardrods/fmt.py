import qimpy as qp
import torch
from mldft1d import Grid1D


class FMT:
    """
    Exact 1D hard rods excess functional. [Percus 1969]
    Calling it FMT, because it is the Fundamental Measure Theory in 1D.
    """

    grid1d: Grid1D
    R: float  #: Radius / half-length `R`
    T: float  #: Temperature
    w0_tilde: torch.Tensor  #: 0-dim weight function (surface measure of FMT in 1D)
    w1_tilde: torch.Tensor  #: 1-dim weight function (volume measure of FMT in 1D)
    w00: float  #: G=0 component of w0_tilde
    w10: float  #: G=0 component of w1_tilde

    def __init__(self, grid1d: Grid1D, *, R: float, T: float) -> None:
        """Initializes to bulk fluid with no external potential."""
        self.grid1d = grid1d
        self.R = R
        self.T = T

        # Initialize FMT weight functions:
        self.w00 = 2  # overall intergal of w0 i.e. G=0 component of w0_tilde
        self.w10 = 2 * R  # overall intergal of w1 i.e. G=0 component of w1_tilde
        self.w0_tilde = self.w00 * (grid1d.Gmag * R).cos()  # FT of w0(x) = delta(R-x)
        self.w1_tilde = self.w10 * (grid1d.Gmag * R).sinc()  # FT of w1(x) = theta(R-x)

    def get_energy(self, n: qp.grid.FieldR) -> torch.Tensor:
        n0 = n.convolve(self.w0_tilde)
        n1 = n.convolve(self.w1_tilde)
        return (-0.5 * self.T) * (n0 ^ (1.0 - n1).log())

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        n0 = n * self.w00
        n1 = n * self.w10
        return (-0.5 * self.T) * (n0 * (1.0 - n1).log())

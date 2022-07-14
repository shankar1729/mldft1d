import qimpy as qp
import numpy as np
import torch
from grid1d import Grid1D


class HardRodsFMT(qp.utils.Minimize[qp.grid.FieldR]):  # type: ignore
    """Exact 1D hard rods functional. [Percus 1969]"""

    grid1d: Grid1D
    R: float  #: Radius / half-length `R`
    T: float  #: Temperature
    n_bulk: float  #: Bulk number density of the fluid
    mu: float  #: Bulk chemical potential
    logn: qp.grid.FieldR  #: State of the classical DFT (effectively local mu)
    V: qp.grid.FieldR  #: External potential
    w0_tilde: qp.grid.FieldR
    w1_tilde: qp.grid.FieldR

    def __init__(self, grid1d: Grid1D, *, R: float, T: float, n_bulk: float) -> None:
        """Initializes to bulk fluid with no external potential."""
        super().__init__(
            comm=qp.rc.comm,
            checkpoint_in=qp.utils.CpPath(),
            name="HardRodsFMT",
            n_iterations=1000,
            energy_threshold=1e-9,
            extra_thresholds={"|grad|": 1e-8},
            method="cg",
        )
        self.grid1d = grid1d
        self.R = R
        self.T = T
        self.set_n_bulk(n_bulk)
        self.logn = qp.grid.FieldR(
            grid1d.grid, data=torch.full_like(grid1d.z, np.log(self.n_bulk))
        )
        self.V = self.logn.zeros_like()

        # Initialize FMT weight functions:
        self.w0_tilde = 2 * (grid1d.Gmag * R).cos()  # F.T. of w0(x) = delta(R-x)
        self.w1_tilde = (2 * R) * (grid1d.Gmag * R).sinc()  # F.T. of w1(x) = theta(R-x)

    @property
    def n(self) -> qp.grid.FieldR:
        """Get current density (from self.logn)"""
        return self.logn.exp()

    def set_n_bulk(self, n_bulk: float) -> None:
        """Update the chemical potential to target given bulk density."""
        self.n_bulk = n_bulk
        # Bulk free energy density, omega(n) = nT(log(n) - log(1-2Rn)) - mu n
        # Derivative domega/dn = T [log(n) - log(1-2Rn) + 1 + 2Rn/(1-2Rn)] - mu = 0
        n1_bulk = (2 * self.R) * n_bulk
        self.mu = self.T * (np.log(n_bulk) - np.log(1 - n1_bulk) + 1 / (1 - n1_bulk))

    def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
        self.logn += step_size * direction

    def compute(self, state, energy_only: bool) -> None:  # type: ignore
        n = self.n
        V_minus_mu = (-self.mu) + self.V
        state.energy["Omega0"] = n ^ (self.T * self.logn + V_minus_mu)  # Ideal-gas part
        # Excess functional:
        n0 = n.convolve(self.w0_tilde)
        n1 = n.convolve(self.w1_tilde)
        state.energy["Fex"] = (-0.5 * self.T) * (n0 ^ (1. - n1).log())
        # Gradient computation:
        if not energy_only:
            n_grad = self.T * (1. + self.logn) + V_minus_mu
            n_grad += (-0.5 * self.T) * (1. - n1).log().convolve(self.w0_tilde)
            n_grad += (+0.5 * self.T) * (n0 / (1. - n1)).convolve(self.w1_tilde)
            state.gradient = qp.grid.FieldR(self.grid1d.grid, data=n_grad.data * n.data)
            state.K_gradient = state.gradient
            state.extra = [state.gradient.norm()]

    def random_direction(self) -> qp.grid.FieldR:
        grid = self.grid1d.grid
        return qp.grid.FieldR(grid, data=torch.randn(grid.shapeR_mine))

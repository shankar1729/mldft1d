import qimpy as qp
import numpy as np
import torch
from grid1d import Grid1D
from nn_function import NNFunction


class MLCDFT(qp.utils.Minimize[qp.grid.FieldR]):  # type: ignore
    """Machine-learned CDFT in 1D."""

    grid1d: Grid1D
    T: float  #: Temperature
    w: NNFunction  #: Weight functions defining spatial nonlocality
    f_ex: NNFunction  #: Free energy density as a function of weighted densities
    n_bulk: float  #: Bulk number density of the fluid
    mu: float  #: Bulk chemical potential
    logn: qp.grid.FieldR  #: State of the classical DFT (effectively local mu)
    V: qp.grid.FieldR  #: External potential

    def __init__(
        self,
        grid1d: Grid1D,
        *,
        T: float,
        w: NNFunction,
        f_ex: NNFunction,
        n_bulk: float
    ) -> None:
        """Initializes to bulk fluid with no external potential."""
        super().__init__(
            comm=qp.rc.comm,
            checkpoint_in=qp.utils.CpPath(),
            name="MLCDFT",
            n_iterations=1000,
            energy_threshold=1e-9,
            extra_thresholds={"|grad|": 1e-8},
            method="cg",
            n_consecutive=1,
        )
        self.grid1d = grid1d
        self.T = T
        self.w = w
        self.f_ex = f_ex
        # Check dimensions:
        assert w.n_in == 1
        assert w.n_out == f_ex.n_in
        assert f_ex.n_out == 1
        self.set_n_bulk(n_bulk)
        self.logn = qp.grid.FieldR(
            grid1d.grid, data=torch.full_like(grid1d.z, np.log(self.n_bulk))
        )
        self.V = self.logn.zeros_like()

    @property
    def n(self) -> qp.grid.FieldR:
        """Get current density (from self.logn)"""
        return self.logn.exp()

    @n.setter
    def n(self, new_n: qp.grid.FieldR) -> None:
        self.logn = new_n.log()

    def set_n_bulk(self, n_bulk: float) -> None:
        """Update the chemical potential to target given bulk density."""
        self.n_bulk = n_bulk
        # Bulk condition: (d/dn) [f_id(n) + f_ex(w * n)] = mu
        w_bulk = self.w(torch.tensor(0.))
        n = torch.tensor(n_bulk)
        n.requires_grad = True
        (self.T * (n * n.log() + self.f_ex(w_bulk * n))).backward()
        self.mu = n.grad.item()

    def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
        self.logn += step_size * direction

    def compute(self, state, energy_only: bool) -> None:  # type: ignore
        n = self.n
        V_minus_mu = (-self.mu) + self.V
        state.energy["Omega0"] = n ^ (self.T * self.logn + V_minus_mu)  # Ideal-gas part
        # Excess functional:
        w_tilde = self.w(self.grid1d.Gmag)
        n_bar = n.convolve(w_tilde)
        state.energy["Fex"] = self.T * self.f_ex(n_bar).integral()
        # Gradient computation:
        if not energy_only:
            n_grad = self.T * (1. + self.logn) + V_minus_mu
            n_bar_grad = self.T * self.f_ex.deriv(n_bar)
            n_grad += n_bar_grad.convolve(w_tilde)
            state.gradient = qp.grid.FieldR(n_grad.grid, data=n_grad.data * n.data)
            state.K_gradient = state.gradient
            state.extra = [state.gradient.norm()]

    def random_direction(self) -> qp.grid.FieldR:
        grid = self.grid1d.grid
        return qp.grid.FieldR(grid, data=torch.randn(grid.shapeR_mine))

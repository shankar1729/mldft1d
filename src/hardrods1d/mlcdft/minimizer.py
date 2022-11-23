import qimpy as qp
import numpy as np
import torch
from ..grid1d import Grid1D
from .functional import Functional


class Minimizer(qp.utils.Minimize[qp.grid.FieldR]):

    functional: Functional  #: Classical DFT free energy functional
    grid1d: Grid1D
    logn: qp.grid.FieldR  #: State of the classical DFT (effectively local mu)
    V: qp.grid.FieldR  #: External potential
    n_bulk: float  #: Bulk number density of the fluid
    mu: float  #: Bulk chemical potential
    T: float  #: Temperature

    def __init__(
        self, *, functional: Functional, grid1d: Grid1D, n_bulk: float
    ) -> None:
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
        self.functional = functional
        self.grid1d = grid1d
        self.n_bulk = n_bulk
        self.mu = functional.get_mu(n_bulk).item()
        self.T = functional.T
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

    def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
        self.logn += step_size * direction

    def compute(self, state, energy_only: bool) -> None:  # type: ignore
        if not energy_only:
            self.logn.data.requires_grad = True
            self.logn.data.grad = None

        n = self.n
        state.energy = self.functional.get_energy(n, self.V - self.mu)

        # Gradient computation:
        if not energy_only:
            state.energy.sum_tensor().backward()  # derivative -> self.logn.data.grad
            state.gradient = qp.grid.FieldR(n.grid, data=self.logn.data.grad)
            state.K_gradient = state.gradient
            state.extra = [state.gradient.norm()]
            self.logn.data.requires_grad = False

    def random_direction(self) -> qp.grid.FieldR:
        grid = self.grid1d.grid
        return qp.grid.FieldR(grid, data=torch.randn(grid.shapeR_mine))

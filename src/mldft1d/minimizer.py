import qimpy as qp
import numpy as np
import torch
from .grid1d import Grid1D
from .protocols import Functional, get_mu


class Minimizer(qp.utils.Minimize[qp.grid.FieldR]):

    functionals: tuple[Functional]  #: Pieces of the DFT functional
    grid1d: Grid1D
    logn: qp.grid.FieldR  #: State of the DFT (effectively local mu)
    n_bulk: float  #: Bulk number density
    mu: float  #: Bulk chemical potential
    V: qp.grid.FieldR  #: External potential
    energy: qp.Energy  #: Equilibrium energy components

    def __init__(
        self,
        *,
        functionals: tuple[Functional],
        grid1d: Grid1D,
        n_bulk: float,
        name: str,
    ) -> None:
        super().__init__(
            comm=qp.rc.comm,
            checkpoint_in=qp.utils.CpPath(),
            name=name,
            n_iterations=1000,
            energy_threshold=1e-9,
            extra_thresholds={"|grad|": 1e-8},
            method="cg",
            n_consecutive=1,
        )
        self.functionals = functionals
        self.grid1d = grid1d
        self.n_bulk = n_bulk
        self.mu = get_mu(functionals, n_bulk)
        self.logn = qp.grid.FieldR(
            grid1d.grid, data=torch.full_like(grid1d.z, np.log(self.n_bulk))
        )
        self.V = self.logn.zeros_like()
        self.energy = qp.Energy()

    @property
    def n(self) -> qp.grid.FieldR:
        """Get current density (from self.logn)"""
        return self.logn.exp()

    @n.setter
    def n(self, n_in: qp.grid.FieldR) -> None:
        self.logn = n_in.log()

    def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
        self.logn += step_size * direction

    def compute(self, state, energy_only: bool) -> None:  # type: ignore
        if not energy_only:
            self.logn.data.requires_grad = True
            self.logn.data.grad = None

        n = self.n
        energy = self.energy
        energy["Ext"] = n ^ (self.V - self.mu)
        for functional in self.functionals:
            energy[functional.__class__.__name__] = functional.get_energy(n)
        state.energy = energy

        # Gradient computation:
        if not energy_only:
            E = state.energy.sum_tensor()
            assert E is not None
            E.backward()  # derivative -> self.logn.data.grad
            state.gradient = qp.grid.FieldR(n.grid, data=self.logn.data.grad)
            state.K_gradient = state.gradient
            state.extra = [state.gradient.norm()]
            self.logn.data.requires_grad = False

    def random_direction(self) -> qp.grid.FieldR:
        grid = self.grid1d.grid
        return qp.grid.FieldR(grid, data=torch.randn(grid.shapeR_mine))

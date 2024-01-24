from __future__ import annotations
from typing import Sequence, Optional, Callable

import torch

from qimpy import rc, Energy
from qimpy.io import CheckpointPath
from qimpy.grid import FieldR
from qimpy.algorithms import Minimize
from .grid1d import Grid1D
from .protocols import Functional, get_mu


class Minimizer(Minimize[FieldR]):

    functionals: Sequence[Functional]  #: Pieces of the DFT functional
    grid1d: Grid1D
    state: FieldR  #: Internal state of the DFT from which density is calculated
    state_to_n: Callable  #: Mapping from independent-variable `state` to density `n`
    n_to_state: Callable  #: Mapping from density `n` to independent-variable `state`
    n_bulk: torch.Tensor  #: Bulk number densities (for each site)
    mu: torch.Tensor  #: Bulk chemical potentials (for each site)
    V: FieldR  #: External potential (for each site)
    energy: Energy  #: Equilibrium energy components

    def __init__(
        self,
        *,
        functionals: Sequence[Functional],
        grid1d: Grid1D,
        n_bulk: torch.Tensor,
        name: str,
        n_iterations: int = 1000,
        energy_threshold: float = 1e-9,
        grad_threshold: float = 1e-8,
        state_to_n: Callable = torch.exp,
        n_to_state: Callable = torch.log,
    ) -> None:
        grid_comm = grid1d.grid.comm
        super().__init__(
            comm=(rc.MPI.COMM_SELF if (grid_comm is None) else grid_comm),
            checkpoint_in=CheckpointPath(),
            name=name,
            n_iterations=n_iterations,
            energy_threshold=energy_threshold,
            extra_thresholds={"|grad|": grad_threshold},
            method="cg",
            n_consecutive=2,
        )
        self.functionals = functionals
        self.grid1d = grid1d
        self.n_bulk = n_bulk
        self.mu = get_mu(functionals, n_bulk)
        self.state = FieldR(
            grid1d.grid,
            data=torch.einsum(
                "s, ... -> s...", n_to_state(n_bulk), torch.ones_like(grid1d.z)
            ),
        )
        self.state_to_n = state_to_n
        self.n_to_state = n_to_state
        self.V = self.state.zeros_like()
        self.energy = Energy()

    @property
    def n(self) -> FieldR:
        """Get current density from `state`"""
        return FieldR(self.grid1d.grid, data=self.state_to_n(self.state.data))

    @n.setter
    def n(self, n_in: FieldR) -> None:
        self.state = FieldR(self.grid1d.grid, data=self.n_to_state(n_in.data))

    def step(self, direction: FieldR, step_size: float) -> None:
        self.state += step_size * direction

    def compute(self, state, energy_only: bool) -> None:  # type: ignore
        V_minus_mu = FieldR(self.V.grid, data=(self.V.data - self.mu.view(-1, 1, 1, 1)))
        if not energy_only:
            self.state.data.requires_grad = True
            self.state.data.grad = None

        n = self.n
        energy = self.energy
        energy["Ext"] = (n ^ V_minus_mu).sum(dim=-1)
        for functional in self.functionals:
            energy[functional.__class__.__name__] = functional.get_energy(n)
        state.energy = energy

        # Gradient computation:
        if not energy_only:
            E = state.energy.sum_tensor()
            assert E is not None
            E.backward()  # derivative -> self.state.data.grad
            state.gradient = FieldR(n.grid, data=self.state.data.grad)
            state.K_gradient = state.gradient
            state.extra = [state.gradient.norm().sum(dim=-1)]
            self.state.data.requires_grad = False

    def random_direction(self) -> FieldR:
        grid = self.grid1d.grid
        return FieldR(grid, data=torch.randn(self.state.data.shape))

    def training_targets(self) -> Optional[tuple[float, FieldR]]:
        """Return energy and potential from only the last term."""
        n = self.n
        n.data.requires_grad = True
        n.data.grad = None
        E = self.functionals[-1].get_energy(n)
        (E / n.grid.dV).backward()  # functional derivative -> n.data.grad
        return E.item(), FieldR(n.grid, data=n.data.grad)

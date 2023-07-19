import qimpy as qp
import torch
from .grid1d import Grid1D
from .protocols import Functional, get_mu
from typing import Sequence, Optional, Callable


class Minimizer(qp.algorithms.Minimize[qp.grid.FieldR]):

    functionals: Sequence[Functional]  #: Pieces of the DFT functional
    grid1d: Grid1D
    state: qp.grid.FieldR  #: Internal state of the DFT from which density is calculated
    state_to_n: Callable  #: Mapping from independent-variable `state` to density `n`
    n_to_state: Callable  #: Mapping from density `n` to independent-variable `state`
    n_bulk: float  #: Bulk number density
    mu: float  #: Bulk chemical potential
    V: qp.grid.FieldR  #: External potential
    energy: qp.Energy  #: Equilibrium energy components

    def __init__(
        self,
        *,
        functionals: Sequence[Functional],
        grid1d: Grid1D,
        n_bulk: float,
        name: str,
        n_iterations: int = 1000,
        energy_threshold: float = 1e-9,
        grad_threshold: float = 1e-8,
        state_to_n: Callable = torch.exp,
        n_to_state: Callable = torch.log,
    ) -> None:
        grid_comm = grid1d.grid.comm
        super().__init__(
            comm=(qp.rc.MPI.COMM_SELF if (grid_comm is None) else grid_comm),
            checkpoint_in=qp.io.CheckpointPath(),
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
        self.state = qp.grid.FieldR(
            grid1d.grid, data=n_to_state(torch.full_like(grid1d.z, self.n_bulk))
        )
        self.state_to_n = state_to_n
        self.n_to_state = n_to_state
        self.V = self.state.zeros_like()
        self.energy = qp.Energy()

    @property
    def n(self) -> qp.grid.FieldR:
        """Get current density from `state`"""
        return qp.grid.FieldR(self.grid1d.grid, data=self.state_to_n(self.state.data))

    @n.setter
    def n(self, n_in: qp.grid.FieldR) -> None:
        self.state = qp.grid.FieldR(self.grid1d.grid, data=self.n_to_state(n_in.data))

    def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
        self.state += step_size * direction

    def compute(self, state, energy_only: bool) -> None:  # type: ignore
        if not energy_only:
            self.state.data.requires_grad = True
            self.state.data.grad = None

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
            E.backward()  # derivative -> self.state.data.grad
            state.gradient = qp.grid.FieldR(n.grid, data=self.state.data.grad)
            state.K_gradient = state.gradient
            state.extra = [state.gradient.norm()]
            self.state.data.requires_grad = False

    def random_direction(self) -> qp.grid.FieldR:
        grid = self.grid1d.grid
        return qp.grid.FieldR(grid, data=torch.randn(grid.shapeR_mine))

    def known_part(self) -> Optional[tuple[float, qp.grid.FieldR]]:
        """Return energy and potential from all but the last term."""
        n = self.n
        n.data.requires_grad = True
        n.data.grad = None
        E = torch.tensor(0.0, device=qp.rc.device)
        for functional in self.functionals[:-1]:
            E += functional.get_energy(n)
        (E / n.grid.dV).backward()  # functional derivative -> n.data.grad
        return E.item(), qp.grid.FieldR(n.grid, data=n.data.grad)

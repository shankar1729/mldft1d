from __future__ import annotations
from typing import Union

from qimpy import MPI
from qimpy.io import CheckpointPath
from qimpy.grid import FieldR
from qimpy.algorithms import Minimize, MinimizeState

from .. import hf


class OEP(Minimize[FieldR]):
    """Optimum Effective Potential using variational minimization."""

    dft: hf.DFT
    Vks: FieldR
    K: float  # Preconditioner

    def __init__(
        self,
        dft: hf.DFT,
        *,
        # comm: MPI.Comm,
        n_iterations: int = 2000,
        energy_threshold: float = 1e-7,
        n_consecutive: int = 1,
        method: str = "l-bfgs",
        cg_type: str = "polak-ribiere",
        line_minimize: str = "auto",
        n_history: int = 15,
        converge_on: Union[str, int] = "all",
    ) -> None:
        self.dft = dft
        super().__init__(
            checkpoint_in=CheckpointPath(),
            comm=MPI.COMM_SELF,
            name="Relax OEP",
            i_iter_start=0,
            n_iterations=n_iterations,
            energy_threshold=energy_threshold,
            extra_thresholds={},
            n_consecutive=n_consecutive,
            method=method,
            cg_type=cg_type,
            line_minimize=line_minimize,
            n_history=n_history,
            converge_on=converge_on,
        )

    def optimize(self):
        self.Vks = self.dft.V + self.dft.n.convolve(self.dft.coulomb_tilde)  # ansatz
        self.minimize()
        self.dft.n.data = self.dft.n.data.detach()

    def step(self, direction: FieldR, step_size: float) -> None:
        """Move the state along `direction` by amount `step_size`"""
        self.Vks += step_size * direction

    def compute(self, state: MinimizeState[FieldR], energy_only: bool) -> None:
        Vks = self.Vks
        if not energy_only:
            Vks.data.requires_grad = True
            Vks.data.grad = None
        dft = self.dft
        dft.C, dft.eig = dft.diagonalize(Vks.data.flatten())
        dft.update(requires_grad=False)
        E = dft.energy.sum_tensor()
        if not energy_only:
            assert E is not None
            E.backward()  # partial derivative dE/dVks -> self.Vks.data.grad
            Vks.data.requires_grad = False
            assert Vks.data.grad is not None
            state.gradient = FieldR(Vks.grid, data=Vks.data.grad)
            state.K_gradient = state.gradient  # preconditioner
        state.energy = dft.energy

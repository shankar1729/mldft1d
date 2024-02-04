from __future__ import annotations
from typing import Union

from qimpy import MPI, Energy
from qimpy.io import CheckpointPath
from qimpy.grid import FieldR
from qimpy.algorithms import Minimize, MinimizeState
from qimpy.profiler import StopWatch

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
        n_iterations: int = 2000,
        energy_threshold: float = 1e-8,
        n_consecutive: int = 2,
        method: str = "l-bfgs",
        cg_type: str = "polak-ribiere",
        line_minimize: str = "auto",
        n_history: int = 100,
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
        dft = self.dft
        if not hasattr(self, "Vks"):
            # Generate initial ansatz:
            self.Vks = dft.V + dft.Vnuc + dft.n.convolve(dft.coulomb_tilde)
        self.minimize()
        dft.n.data = dft.n.data.detach()

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
            watch = StopWatch("OEP.Vks_grad")
            E.backward()  # partial derivative dE/dVks -> self.Vks.data.grad
            watch.stop()
            Vks.data.requires_grad = False
            assert Vks.data.grad is not None
            state.gradient = FieldR(Vks.grid, data=Vks.data.grad)
            state.K_gradient = state.gradient
        state.energy = dft.energy

    def perturb_n_electrons(self, delta_n_electrons: float) -> Energy:
        """Return energy changes due to finite perturbation of electron count.
        Restore state to unperturbed state after calculation."""
        # Backup current state:
        dft = self.dft
        energy0 = Energy({name: value.item() for name, value in dft.energy.items()})
        Vks0 = self.Vks.clone()

        # Perturb:
        dft.n_electrons += delta_n_electrons
        self.optimize()
        denergy = Energy(
            {name: (value.item() - energy0[name]) for name, value in dft.energy.items()}
        )

        # Restore original state:
        dft.n_electrons -= delta_n_electrons
        self.Vks = Vks0
        self.compute(MinimizeState[FieldR](), energy_only=True)
        return denergy

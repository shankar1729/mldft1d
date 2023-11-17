from __future__ import annotations
from typing import Optional, Sequence, Union

import numpy as np
import torch

from qimpy import Energy, log, MPI
from qimpy.algorithms import Minimize, MinimizeState

from .. import hf


class OEP(Minimize[torch.Tensor]):
    """Class to find optimized effective potential through a variational minimization
    of Kohn-Sham potential
    """

    dft: hf.DFT
    Vks: torch.Tensor
    K: float  # Preconditioner

    def __init__(
        self,
        dft: hf.DFT,
        *,
        # comm: MPI.Comm,
        n_iterations: int = 20,
        energy_threshold: float = 5e-5,
        fmax_threshold: float = 5e-4,
        n_consecutive: int = 1,
        method: str = "l-bfgs",
        cg_type: str = "polak-ribiere",
        line_minimize: str = "auto",
        n_history: int = 15,
        converge_on: Union[str, int] = "all",
    ) -> None:
        self.dft = dft
        extra_thresholds = {"fmax": fmax_threshold}
        super().__init__(
            checkpoint_in=None,
            comm=MPI.COMM_SELF,
            name="Relax OEP",
            i_iter_start=0,
            n_iterations=n_iterations,
            energy_threshold=energy_threshold,
            extra_thresholds=extra_thresholds,
            n_consecutive=n_consecutive,
            method=method,
            cg_type=cg_type,
            line_minimize=line_minimize,
            n_history=n_history,
            converge_on=converge_on,
        )

    def optimize(self):
        ansatz = self.dft.V + 0.5 * self.dft.n.convolve(self.dft.coulomb_tilde)
        self.Vks = ansatz.data  # Vks ansatz
        print("Vks shape ", self.Vks.shape)
        E, gradient = self.compute_energy()
        self.finite_difference_test(gradient)
        # self.minimize()

    def step(self, direction: torch.Tensor, step_size: float) -> None:
        """Move the state along `direction` by amount `step_size`"""
        self.Vks += step_size * direction

    def compute(self, state: MinimizeState[torch.Tensor], energy_only: bool) -> None:
        state.energy, state.gradient = self.compute_energy()

    def compute_energy(self) -> tuple[Energy, torch.Tensor]:
        Vks = self.Vks.detach()
        Vks.requires_grad = True
        Vks.grad = None
        dft = self.dft
        dft.C, dft.eig = dft.diagonalize(Vks)
        dft.update()
        E = dft.energy.sum_tensor()
        assert E is not None
        E.backward()  # partial derivative dE/dVks -> self.Vks.data.grad
        Vks.requires_grad = False
        return dft.energy, Vks.grad.flatten().detach() / dft.grid1d.grid.dV

    def finite_difference_test(
        self, direction: torch.Tensor, step_sizes: Optional[Sequence[float]] = None
    ) -> None:
        """Check gradient implementation by taking steps along `direction`.
        This will print ratio of actual energy differences along steps of
        various sizes in `step_sizes` and the expected energy difference
        based on the gradient. A correct implementation should show a ratio
        approaching 1 for a range of step sizes, with deviations at lower
        step sizes due to round off error and at higher step sizes due to
        nonlinearity."""
        log.info(f'{self.name}: {"-" * 12} Finite difference test {"-" * 12}')
        if step_sizes is None:
            step_sizes = np.logspace(-9, 1, 11).tolist()
        # Initial state with gradient:
        state = MinimizeState["torch.Tensor"]()
        E0 = self._compute(state, energy_only=False)
        dE_step = self._sync(
            state.gradient.vdot(direction)
        )  # directional derivative along step direction
        # Finite difference derivatives:
        step_size_prev = 0.0  # cumulative progress along step:
        for step_size in sorted(step_sizes):
            self.step(direction, step_size - step_size_prev)
            step_size_prev = step_size
            deltaE = self._compute(state, energy_only=True) - E0
            dE_expected = dE_step * step_size
            log.info(
                f"{self.name}: step size: {step_size:.3e}"
                f"  d{state.energy.name}"
                f" ratio: {deltaE / dE_expected:.11f}"
            )
        log.info(f'{self.name}: {"-" * 48}')
        # Restore original position:
        self.step(direction, -step_size_prev)

from __future__ import annotations
from typing import Union

import torch

from qimpy import MPI
from qimpy.io import CheckpointPath
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
        ansatz = self.dft.V.data + self.dft.n.convolve(self.dft.coulomb_tilde).data
        self.Vks = ansatz.flatten()  # Vks ansatz
        self.minimize()
        self.dft.n.data = self.dft.n.data.detach()

    def step(self, direction: torch.Tensor, step_size: float) -> None:
        """Move the state along `direction` by amount `step_size`"""
        self.Vks += step_size * direction

    def compute(self, state: MinimizeState[torch.Tensor], energy_only: bool) -> None:
        Vks = self.Vks
        if not energy_only:
            Vks.requires_grad = True
            Vks.grad = None
        dft = self.dft
        dft.C, dft.eig = dft.diagonalize(Vks)
        dft.update(requires_grad=False)
        E = dft.energy.sum_tensor()
        if not energy_only:
            assert E is not None
            E.backward()  # partial derivative dE/dVks -> self.Vks.data.grad
            Vks.requires_grad = False
            assert Vks.grad is not None
            state.gradient = Vks.grad
            state.K_gradient = Vks.grad  # preconditioner
        state.energy = dft.energy

    """
    def finite_difference_test(
        self, direction: torch.Tensor, step_sizes: Optional[Sequence[float]] = None
    ) -> None:
        '''Check gradient implementation by taking steps along `direction`.
        This will print ratio of actual energy differences along steps of
        various sizes in `step_sizes` and the expected energy difference
        based on the gradient. A correct implementation should show a ratio
        approaching 1 for a range of step sizes, with deviations at lower
        step sizes due to round off error and at higher step sizes due to
        nonlinearity.'''
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
    """

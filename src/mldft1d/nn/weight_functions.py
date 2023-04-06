from __future__ import annotations
from typing import Protocol, Optional
import qimpy as qp
import numpy as np
import functools
import torch
from .function import Function


class Normalizer:
    """Wrapper to produce orthonormal weight functions with correct symmetries."""

    rc: float  #: If nonzero, compute weight function in real space with this cutoff
    Gc: float  #: If nonzero, smoothly cutoff reciprocal space w beyond Gc (in a0^-1)
    calc0: Calculator  #: Common calculator used for orthonormalization

    def __init__(self, rc: float, Gc: float, L0: float = 100.0, Nz0: int = 500) -> None:
        """Prepare to calculate real/reciprocal space weights based on rc and Gc.
        Note that L0 and Nz0 are for a common grid used to normalize weights,
        and should remain fixed for a Functional throughout its training and usage.
        The actual grid used for evaluating the weight functions are specified later.
        """
        self.rc = rc
        self.Gc = Gc
        self.calc0 = get_calculator(L0, Nz0, rc, Gc)

    def get_weights(
        self, grid: qp.grid.Grid, w: Function, n_weights_even: int
    ) -> torch.Tensor:
        """Get weight functions `w` on `grid`, with `n_weights_even` of them even."""
        ortho = self._get_weights_ortho(w, n_weights_even)[1]
        L = grid.lattice.Rbasis[2, 2].item()
        Nz = grid.shape[2]
        return ortho @ get_calculator(L, Nz, self.rc, self.Gc)(w, n_weights_even)

    def get_weights_Gzero(self, w: Function, n_weights_even: int) -> torch.Tensor:
        """Corresponding to `get_weights`, but only get G=0 component.
        The result has dimensions `n_weights_even`, since the odd weights
        have zero G=0 component."""
        w_tilde, ortho = self._get_weights_ortho(w, n_weights_even)
        return (ortho @ w_tilde)[:n_weights_even, 0].real

    def _get_weights_ortho(
        self, w: Function, n_weights_even: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_tilde = self.calc0(w, n_weights_even)
        Gweight = torch.full(
            w_tilde.shape[1:], 2.0, dtype=w_tilde.dtype, device=qp.rc.device
        )
        Gweight[0] = 1.0
        overlap = torch.einsum("iG, jG, G -> ij", w_tilde.conj(), w_tilde, Gweight).real
        ortho = qp.utils.ortho_matrix(overlap).to(w_tilde.dtype).T
        return w_tilde, ortho


class Calculator(Protocol):
    """Underlying weight-function calculator."""

    def __call__(self, w: Function, n_weights_even: int) -> torch.Tensor:
        ...


@functools.cache
def get_calculator(L: float, Nz: int, rc: float, Gc: float) -> Calculator:
    if rc:
        return CalculatorR(L, Nz, rc)
    else:
        return CalculatorG(L, Nz, Gc)


class CalculatorG:
    """Reciprocal-space weight calculator."""

    Gsq: torch.Tensor  #: reciprocal lattice vector squares (input to NN)
    gradient_z: torch.Tensor  #: gradient operator
    cutoff: Optional[torch.Tensor]  #: optional reciprocal space cutoff (if non-zero)

    def __init__(self, L: float, Nz: int, Gc: float) -> None:
        Nz_half = Nz // 2 + 1  # real-FFT dimensions
        G = (2 * np.pi / L) * torch.arange(Nz_half, device=qp.rc.device)[None]
        self.Gsq = G.square()
        self.gradient_z = 1j * G
        self.cutoff = None
        if Gc:
            sel = torch.where(G < Gc)
            self.cutoff = torch.zeros_like(self.Gsq)
            self.cutoff[sel] = (((np.pi / Gc) * G[sel]).cos() + 1.0) * 0.5

    def __call__(self, w: Function, n_weights_even: int) -> torch.Tensor:
        w_tilde = w(self.Gsq)
        if self.cutoff is not None:
            w_tilde *= self.cutoff

        # Switch symmetries to odd for some of the weight functions:
        w_tilde = w_tilde.to(torch.complex128)
        w_tilde[n_weights_even:] *= self.gradient_z
        return w_tilde


class CalculatorR:
    """Real-space weight calculator."""

    sup: int  #: supercell multiplier to accommodate range of kernel
    dz: float  #: grid spacing
    z_sup: torch.Tensor  #: z in supercell
    z_sup_sq: torch.Tensor  #: z^2 in supercell (input to NN weight function)
    prefactor: torch.Tensor  #: weight function prefactor enforcing smooth falloff at rc

    def __init__(self, L: float, Nz: int, rc: float) -> None:
        # Find supercell needed to accommodate rc:
        sup = int(np.ceil(2 * rc / L))
        Nz_sup = Nz * sup
        iz_sup = torch.arange(Nz_sup, device=qp.rc.device)[None]  # extra dim for bcast
        dz = L / Nz
        z_sup = dz * torch.where(iz_sup <= Nz_sup // 2, iz_sup, iz_sup - Nz_sup)

        # Store required quantities:
        self.sup = sup
        self.dz = dz
        self.z_sup = z_sup
        self.z_sup_sq = z_sup.square()
        self.prefactor = torch.where(
            abs(z_sup) < rc, 1.0 + torch.cos(z_sup * np.pi / rc), 0.0
        )  # value and derivative are both zero at rc

    def __call__(self, w: Function, n_weights_even: int) -> torch.Tensor:
        # Compute in real space (enforce positive, bounded and cutoff):
        w_real = self.prefactor / (1.0 + w(self.z_sup_sq).square())

        # Switch symmetries to odd for some of the weight functions:
        w_real[n_weights_even:] *= self.z_sup

        # Convert to half reciprocal space and downselect to orginal cell:
        return (torch.fft.rfft(w_real) * self.dz)[:, :: self.sup]

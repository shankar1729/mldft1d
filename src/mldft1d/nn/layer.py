from __future__ import annotations
from functools import cache

import torch

from qimpy.grid import Grid, FieldR
from qimpy.io.dict import key_cleanup
from .weight_functions import WeightFunctions, make_weight_functions


class Layer(torch.nn.Module):  # type: ignore
    """One layer of a machine-learned DFT in 1D."""

    n_in: tuple[int, int]  #: Number of even and odd input channels to layer
    n_out: tuple[int, int]  #: Number of even and odd output channels from layer
    weight_functions: WeightFunctions  #: Trainable weight functions

    def __init__(
        self,
        *,
        n_in: tuple[int, int],
        n_out: tuple[int, int],
        weight_functions: dict,
    ) -> None:
        """Initializes functional with specified sizes (and random parameters)."""
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.weight_functions = make_weight_functions(
            **key_cleanup(weight_functions),
            n_functions=sum(n_in) * sum(n_out),
        )

    def asdict(self) -> dict:
        """Serialize parameters to dict."""
        return dict(
            n_in=self.n_in,
            n_out=self.n_out,
            weight_functions=self.weight_functions.asdict(),
        )

    def get_w_tilde(self, grid: Grid, n_dim_tot: int = 2) -> torch.Tensor:
        """Compute weights for specified grid and total dimension count.
        Optionally, suppress local/gradient contributions for plotting.
        """
        Gz = self.Gz(grid)
        w_tilde = self.weight_functions(Gz).unflatten(0, (sum(self.n_in), -1))
        w_tilde = w_tilde.to(torch.complex128)  # to accommodate odd weights

        # Add gradient term to make odd weight functions odd:
        n_in_even = self.n_in[0]
        n_out_even = self.n_out[0]
        iGz = 1j * Gz
        w_tilde[:n_in_even, n_out_even:] *= iGz  # even in, odd out
        w_tilde[n_in_even:, :n_out_even] *= iGz  # odd in, even out
        return w_tilde.unflatten(-1, (1,) * (n_dim_tot - 3) + (-1,))  # Singleton dims

    def compute(self, n: FieldR) -> FieldR:
        w_tilde = self.get_w_tilde(n.grid, len(n.data.shape) + 1)
        return n.convolve(w_tilde, "i..., iw... -> w...")

    @cache
    def Gz(self, grid: Grid) -> torch.Tensor:
        """Get reciprocal lattice vectors from grid.
        Cached to ensure same Gz object returned for a given grid, which allows
        G-dependent constants to be cached during training and inference."""
        return grid.get_gradient_operator("H")[2, 0, 0].imag

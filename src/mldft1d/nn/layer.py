from __future__ import annotations
from functools import cache

import torch
from torch.nn.parameter import Parameter
import qimpy as qp
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
            n_functions=sum(n_out),
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
        w_tilde = self.weight_functions(Gz).unflatten(0, (sum(self.n_out), -1))
        w_tilde = w_tilde.to(torch.complex128)  # to accommodate odd weights

        # Add gradient term to make odd weight functions odd:
        n_out_even = self.n_out[0]
        iGz = 1j * Gz
        w_tilde[n_out_even:] *= iGz  # second half of vector is odd
        return w_tilde.unflatten(-1, (1,) * (n_dim_tot - 3) + (-1,))  # Singleton dims

    def compute(self, n: FieldR) -> FieldR:
        w_tilde = self.get_w_tilde(n.grid, len(n.data.shape) + 1)
        conv_ab = n.convolve(w_tilde, "i..., w... -> iw...")
        coeff_reduce = torch.Tensor(
            Parameter(torch.empty((sum(self.n_in),), device=qp.rc.device))
        )
        # check1 = n.convolve(w_tilde, "i..., iw... -> w...")
        # check2 = conv_ab.dot(coeff_reduce)
        # print("CHECKS:", type(check1), type(check2), type(w_tilde), type(conv_ab), type(coeff_reduce))
        return torch.einsum("i...,iw... -> w...", coeff_reduce, conv_ab.data)
        # return conv_ab.dot(coeff_reduce)
        # return n.convolve(w_tilde, "i..., iw... -> w...")

    @cache
    def Gz(self, grid: Grid) -> torch.Tensor:
        """Get reciprocal lattice vectors from grid.
        Cached to ensure same Gz object returned for a given grid, which allows
        G-dependent constants to be cached during training and inference."""
        return grid.get_gradient_operator("H")[2, 0, 0].imag

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
    n_weights: tuple[int, int]  #: Number of even and odd weight functions in layer
    weight_functions: WeightFunctions  #: Trainable weight functions
    Weee: torch.Tensor
    Weoo: torch.Tensor
    Wooe: torch.Tensor
    Woeo: torch.Tensor

    def __init__(
        self,
        *,
        n_in: tuple[int, int],
        n_out: tuple[int, int],
        n_weights: tuple[int, int],
        weight_functions: dict,
    ) -> None:
        """Initializes functional with specified sizes (and random parameters)."""
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_weights = n_weights
        self.weight_functions = make_weight_functions(
            **key_cleanup(weight_functions),
            n_functions=sum(n_weights),
        )
        self.Weee = Parameter(
            torch.empty((n_out[0], n_weights[0], n_in[0]), device=qp.rc.device)
        )
        self.Weoo = Parameter(
            torch.empty((n_out[0], n_weights[1], n_in[1]), device=qp.rc.device)
        )
        self.Wooe = Parameter(
            torch.empty((n_out[1], n_weights[1], n_in[0]), device=qp.rc.device)
        )
        self.Woeo = Parameter(
            torch.empty((n_out[1], n_weights[0], n_in[1]), device=qp.rc.device)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.Weee, -1.0, 1.0)
        torch.nn.init.uniform_(self.Weoo, -1.0, 1.0)
        torch.nn.init.uniform_(self.Woeo, -1.0, 1.0)
        torch.nn.init.uniform_(self.Wooe, -1.0, 1.0)

    def asdict(self) -> dict:
        """Serialize parameters to dict."""
        return dict(
            n_in=self.n_in,
            n_out=self.n_out,
            n_weights=self.n_weights,
            weight_functions=self.weight_functions.asdict(),
        )

    def get_w_tilde(self, grid: Grid, n_dim_tot: int = 2) -> torch.Tensor:
        """Compute weights for specified grid and total dimension count.
        Optionally, suppress local/gradient contributions for plotting.
        """
        Gz = self.Gz(grid)
        w_tilde = self.weight_functions(Gz).to(
            torch.complex128
        )  # to accommodate odd weights

        # Add gradient term to make odd weight functions odd:
        n_weights_even = self.n_weights[0]
        iGz = 1j * Gz
        w_tilde[n_weights_even:] *= iGz  # second half of vector is odd
        return w_tilde.unflatten(-1, (1,) * (n_dim_tot - 2) + (-1,))  # Singleton dims

    def compute(self, n: FieldR) -> FieldR:
        n_out = self.n_out
        n_in = self.n_in
        n_weights = self.n_weights
        w_tilde = self.get_w_tilde(n.grid, len(n.data.shape))
        conv_ab = n.convolve(w_tilde, "i..., w... -> iw...")
        W = torch.zeros((sum(n_out), sum(n_weights), sum(n_in)))
        W[: n_out[0], : n_weights[0], : n_in[0]] = self.Weee
        W[: n_out[0], n_weights[0] :, n_in[0] :] = self.Weoo
        W[n_out[0] :, n_weights[0] :, : n_in[0]] = self.Wooe
        W[n_out[0] :, : n_weights[0], n_in[0] :] = self.Woeo
        return FieldR(n.grid, data=torch.einsum("owi,iw... -> o...", W, conv_ab.data))

    @cache
    def Gz(self, grid: Grid) -> torch.Tensor:
        """Get reciprocal lattice vectors from grid.
        Cached to ensure same Gz object returned for a given grid, which allows
        G-dependent constants to be cached during training and inference."""
        return grid.get_gradient_operator("H")[2, 0, 0].imag

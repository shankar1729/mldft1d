from __future__ import annotations
import qimpy as qp
import torch
from functools import cache
from .weight_functions import WeightFunctions, make_weight_functions
from .function import Function, Linear
from typing import Optional


class Layer(torch.nn.Module):  # type: ignore
    """One layer of a machine-learned DFT in 1D."""

    n_weights: tuple[int, int]  #: Number of even and odd weight functions
    use_local: bool  #: If true, local density is the final even weighted density
    use_gradient: bool  #: If true, local gradient is the final odd weighted density
    n_local: tuple[int, int]  #: Number of local even and odd weight functions
    n_nonlocal: tuple[int, int]  #: Number of nonlocal even and odd weight functions
    i_odd_pair: torch.Tensor  #: Indices of distinct pairs of odd weighted densities
    n_irred: int  #: Number of irreducible scalar combinations of weighted densities
    n_inputs: int  #: Number of input channels to layer
    coefficients_local: Optional[Linear]  #: Linear combination of local input/gradients
    weight_functions: WeightFunctions  #: Trainable weight functions
    f: Function  #: Per-particle free energy function of weighted densities

    def __init__(
        self,
        *,
        n_weights: tuple[int, int],
        use_local: bool = False,
        use_gradient: bool = False,
        n_inputs: int,
        n_outputs: int,
        weight_functions: dict,
        hidden_sizes: list[int],
        activation: str = "softplus",
    ) -> None:
        """Initializes functional with specified sizes (and random parameters)."""
        super().__init__()
        n_weights_even, n_weights_odd = n_weights
        self.n_weights = n_weights
        self.use_local = use_local
        self.use_gradient = use_gradient
        self.n_local = (1 if use_local else 0, 1 if use_gradient else 0)
        self.n_nonlocal = (
            n_weights_even - self.n_local[0],
            n_weights_odd - self.n_local[1],
        )
        self.i_odd_pair = get_pair_indices(n_weights_odd)
        self.n_irred = n_weights_even + len(self.i_odd_pair)
        self.n_inputs = n_inputs
        self.weight_functions = make_weight_functions(
            **qp.utils.dict.key_cleanup(weight_functions),
            n_functions=n_inputs * sum(self.n_nonlocal),
        )
        self.coefficients_local = (
            Linear(1, n_inputs * sum(self.n_local), bias=False, device=qp.rc.device)
            if (use_local or use_gradient)
            else None
        )
        self.f = Function(self.n_irred, n_outputs, hidden_sizes, activation)

    def asdict(self) -> dict:
        """Serialize parameters to dict."""
        return dict(
            n_weights=self.n_weights,
            use_local=self.use_local,
            use_gradient=self.use_gradient,
            n_inputs=self.n_inputs,
            n_outputs=self.f.n_out,
            weight_functions=self.weight_functions.asdict(),
            hidden_sizes=self.f.n_hidden,
            activation=self.f.activation.__class__.__name__.lower(),
        )

    def get_w_tilde(self, grid: qp.grid.Grid, n_dim_tot: int = 2) -> torch.Tensor:
        """Compute weights for specified grid and total dimension count.
        Optionally, suppress local/gradient contributions for plotting.
        """
        Gz = self.Gz(grid)
        w_tilde = self.weight_functions(Gz).unflatten(0, (self.n_inputs, -1))
        if self.coefficients_local is not None:
            ones = torch.ones_like(Gz)[None]
            w_locals = self.coefficients_local(ones).unflatten(0, (self.n_inputs, -1))
            w_local, w_gradient = w_locals.split(self.n_local, dim=1)

            # Reconstitute w_tildes to include local pieces:
            w_even, w_odd = w_tilde.split(self.n_nonlocal, dim=1)
            w_tilde = torch.cat((w_even, w_local, w_odd, w_gradient), dim=1)

        # Add gradient term to make odd weight functions odd:
        w_tilde = w_tilde.to(torch.complex128)
        w_tilde[:, self.n_weights[0] :] *= 1j * Gz
        return w_tilde.unflatten(-1, (1,) * (n_dim_tot - 3) + (-1,))  # Singleton dims

    def get_w_zero_even(self) -> torch.Tensor:
        """Get G=0 component of the even weight functions.
        (The odd weight functions have zero G=0 component by definition."""
        Gzero = torch.tensor(0.0, device=qp.rc.device)
        w_tilde = self.weight_functions(Gzero).unflatten(0, (self.n_inputs, -1))
        w_even, _ = w_tilde.split(self.n_nonlocal, dim=1)
        if (self.coefficients_local is not None) and self.use_local:
            ones = torch.ones_like(Gzero)[None]
            w_locals = self.coefficients_local(ones).unflatten(0, (self.n_inputs, -1))
            w_local, _ = w_locals.split(self.n_local, dim=1)
            w_even = torch.cat((w_even, w_local), dim=1)
        return w_even

    def compute(self, n: qp.grid.FieldR) -> qp.grid.FieldR:
        w_tilde = self.get_w_tilde(n.grid, len(n.data.shape) + 1)
        n_bar = n.convolve(w_tilde, "i..., iw... -> w...")
        n_even, n_odd = n_bar.data.split(self.n_weights)
        n_odd_sq = (n_odd[:, None] * n_odd[None]).flatten(0, 1)[self.i_odd_pair]
        scalars = torch.cat((n_even, n_odd_sq), dim=0)
        return qp.grid.FieldR(n.grid, data=self.f(scalars))

    def compute_bulk(self, n: torch.Tensor) -> torch.Tensor:
        n_even = torch.einsum("i..., iw -> w...", n, self.get_w_zero_even())
        n_odd_sq = torch.zeros(
            (len(self.i_odd_pair),) + n.shape[1:], device=n.device, dtype=n.dtype
        )
        scalars = torch.cat((n_even, n_odd_sq), dim=0)
        return self.f(scalars)

    @cache
    def Gz(self, grid: qp.grid.Grid) -> torch.Tensor:
        """Get reciprocal lattice vectors from grid.
        Cached to ensure same Gz object returned for a given grid, which allows
        G-dependent constants to be cached during training and inference."""
        return grid.get_gradient_operator("H")[2, 0, 0].imag


def get_pair_indices(N: int) -> torch.Tensor:
    """Get indices of distinct unordered pairs (i, j) for i, j in [0, N).
    The indices are to the flattened array of all pairs, i.e. in [0, N^2),
    (and will have length N(N+1)/2)."""
    index = torch.arange(N, dtype=torch.long, device=qp.rc.device)
    i, j = torch.where(index[:, None] <= index[None, :])
    return i * N + j

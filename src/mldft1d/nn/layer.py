from __future__ import annotations
import qimpy as qp
import torch
from .basis import Basis, make_basis
from .function import Function, Linear


class Layer(torch.nn.Module):  # type: ignore
    """One layer of a machine-learned DFT in 1D."""

    n_weights: tuple[int, int]  #: Number of even and odd weight functions
    i_odd_pair: torch.Tensor  #: Indices of distinct pairs of odd weighted densities
    n_irred: int  #: Number of irreducible scalar combinations of weighted densities
    basis: Basis  #: Basis functions for expanding the nonlocal weight functions
    n_inputs: int  #: Number of input channels to layer
    w: torch.nn.ModuleList  #: Linear combinations of even and odd basis
    f: Function  #: Per-particle free energy function of weighted densities

    def __init__(
        self,
        *,
        n_weights: tuple[int, int],
        basis: dict,
        n_inputs: int,
        n_outputs: int,
        hidden_sizes: list[int],
        activation: str = "softplus",
    ) -> None:
        """Initializes functional with specified sizes (and random parameters)."""
        super().__init__()
        n_weights_even, n_weights_odd = n_weights
        self.n_weights = n_weights
        self.i_odd_pair = get_pair_indices(n_weights_odd)
        self.n_irred = n_weights_even + len(self.i_odd_pair)
        self.basis = make_basis(**basis)
        self.n_inputs = n_inputs
        self.w = torch.nn.ModuleList(
            [
                Linear(
                    n_basis_i, n_inputs * n_weights_i, bias=False, device=qp.rc.device
                )
                for n_weights_i, n_basis_i in zip(
                    n_weights, (self.basis.n_basis_even, self.basis.n_basis_odd)
                )
            ]
        )
        self.f = Function(self.n_irred, n_outputs, hidden_sizes, activation)

    def asdict(self) -> dict:
        """Save parameters to specified filename."""
        return dict(
            n_weights=self.n_weights,
            basis=self.basis.asdict(),
            n_inputs=self.n_inputs,
            n_outputs=self.f.n_out,
            hidden_sizes=self.f.n_hidden,
            activation=self.f.activation.__class__.__name__.lower(),
        )

    def get_w_tilde(
        self, grid: qp.grid.Grid, n_dim_tot: int = 2, suppress_local: bool = False
    ) -> torch.Tensor:
        """Compute weights for specified grid and total dimension count.
        Optionally, suppress local/gradient contributions for plotting.
        """
        w_tildes = [
            w(basis).unflatten(0, (self.n_inputs, -1))
            for w, basis in zip(self.w, self.basis(grid, suppress_local))
        ]
        w_tilde = torch.cat(w_tildes, dim=1)  # combine for efficient convolution below
        return w_tilde.unflatten(-1, (1,) * (n_dim_tot - 3) + (-1,))  # Singleton dims

    def get(self, n: qp.grid.FieldR) -> qp.grid.FieldR:
        w_tilde = self.get_w_tilde(n.grid, len(n.data.shape) + 1)
        n_bar = n.convolve(w_tilde, "i..., iw... -> w...")
        n_even, n_odd = n_bar.data.split(self.n_weights)
        n_odd_sq = (n_odd[:, None] * n_odd[None]).flatten(0, 1)[self.i_odd_pair]
        scalars = qp.grid.FieldR(n.grid, data=torch.cat((n_even, n_odd_sq), dim=0))
        return qp.grid.FieldR(n.grid, data=self.f(scalars.data))

    def get_bulk(self, n: torch.Tensor) -> torch.Tensor:
        basis_o_even = self.basis.o[0].real  # G=0 component of even basis (0 for odd)
        w_o_even = self.w[0](basis_o_even).unflatten(0, (self.n_inputs, -1))
        n_even = torch.einsum("i..., iw -> w...", n, w_o_even)
        n_odd_sq = torch.zeros(
            (len(self.i_odd_pair),) + n.shape[1:], device=n.device, dtype=n.dtype
        )
        scalars = torch.cat((n_even, n_odd_sq), dim=0)
        return self.f(scalars)


def get_pair_indices(N: int) -> torch.Tensor:
    """Get indices of distinct unordered pairs (i, j) for i, j in [0, N).
    The indices are to the flattened array of all pairs, i.e. in [0, N^2),
    (and will have length N(N+1)/2)."""
    index = torch.arange(N, dtype=torch.long, device=qp.rc.device)
    i, j = torch.where(index[:, None] <= index[None, :])
    return i * N + j

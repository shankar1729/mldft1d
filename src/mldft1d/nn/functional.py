from __future__ import annotations
import qimpy as qp
import torch
import os
from mpi4py import MPI
from typing import Optional
from .function import Function


class Functional(torch.nn.Module):  # type: ignore
    """Machine-learned DFT in 1D."""

    n_weights: tuple[int, int]  #: Number of even and odd weight functions
    i_odd_pair: torch.Tensor  #: Indices of distinct pairs of odd weighted densities
    n_irred: int  #: Number of irreducible scalar combinations of weighted densities
    w: Function  #: Weight functions defining spatial nonlocality
    f: Function  #: Per-particle free energy function of weighted densities
    Gc: float  #: Reciprocal space cutoff on weight functions
    G_sq_power: torch.Tensor  #: Power of G^2 associated with each weight function
    G_sq_norm: torch.Tensor  #: Normalizing factor for spectral power in each weight
    cache_w: bool  #: Whether to precompute weight functions (must be off for training)

    _cached_w_tilde: dict[qp.grid.Grid, torch.Tensor]  #: cached weight functions
    _cached_w0: Optional[torch.Tensor]  #: cached G=0 component of weights

    def __init__(
        self,
        comm: MPI.Comm,
        *,
        n_weights: tuple[int, int],
        w_hidden_sizes: list[int],
        f_hidden_sizes: list[int],
        Gc: float,
        cache_w: bool = True,
    ) -> None:
        """Initializes functional with specified sizes (and random parameters)."""
        super().__init__()
        n_weights_even, n_weights_odd = n_weights
        self.n_weights = n_weights
        self.i_odd_pair = get_pair_indices(n_weights_odd)
        self.n_irred = n_weights_even + len(self.i_odd_pair)
        self.w = Function(1, sum(n_weights), w_hidden_sizes)
        self.f = Function(self.n_irred, self.n_irred, f_hidden_sizes)
        self.Gc = Gc
        self.G_sq_power = torch.cat(
            (
                torch.arange(n_weights_even, dtype=torch.int),
                torch.arange(n_weights_odd, dtype=torch.int),
            )
        ).to(qp.rc.device)[:, None]
        G_power = 2 * self.G_sq_power
        G_power[n_weights_even:] += 1
        G_norm_terms = 2 * G_power + torch.arange(1, 10, 2, device=qp.rc.device)[None]
        self.G_sq_norm = (G_norm_terms.prod(dim=1) / 384).sqrt()[:, None]
        self.G_sq_norm[n_weights_even:] *= 1.0 / Gc
        self.cache_w = cache_w
        if comm.size > 1:
            self.bcast_parameters(comm)
        self._cached_w_tilde = dict()
        self._cached_w0 = None

    @classmethod
    def load(
        cls,
        comm: MPI.Comm,
        *,
        load_file: str = "",
        **kwargs,
    ) -> Functional:
        """
        Initialize functional from `kwargs` or from params file if given `load_file`.
        Any parameter in `kwargs` must be consistent with the params file, if given.
        """
        params = dict(**kwargs)
        state = {}

        # Merge / check parameters from file, if available:
        if load_file and os.path.isfile(load_file):
            params_in = torch.load(load_file, map_location=qp.rc.device)
            for key, value in params_in.items():
                if key == "state":
                    state = value
                elif key in params:
                    assert params[key] == value
                else:
                    params[key] = value

        # Create functional and load state if available:
        functional = Functional(comm, **params)
        if state:
            functional.load_state_dict(state)
        return functional

    def save(self, filename: str, comm: MPI.Comm) -> None:
        """Save parameters to specified filename."""
        params = dict(
            n_weights=self.n_weights,
            w_hidden_sizes=self.w.n_hidden,
            f_hidden_sizes=self.f.n_hidden,
            Gc=self.Gc,
            state=self.state_dict(),
        )
        if comm.rank == 0:
            torch.save(params, filename)

    def get_w_tilde(self, grid: qp.grid.Grid, n_dim_tot: int = 2) -> torch.Tensor:
        """Compute weights for specified grid and total dimension count."""
        if self.cache_w and (grid in self._cached_w_tilde):
            w_tilde = self._cached_w_tilde[grid]
        else:
            # Compute weight:
            gradient_z = grid.get_gradient_operator("H")[2, 0]  # 1 x Nz array
            g_sq = torch.clamp(qp.utils.abs_squared(gradient_z / self.Gc), max=1.0)
            w_tilde = (
                self.w(g_sq)  # Neural network part
                * (1.0 - g_sq).square()  # Cutoff at Gc
                * (g_sq**self.G_sq_power)  # Different G -> 0 power for each weight
                * self.G_sq_norm  # Ensure similar spectral power in each weight
            )

            # Make odd weights:
            w_tilde = w_tilde.to(torch.complex128)
            w_tilde[self.n_weights[0] :] *= gradient_z

            if self.cache_w:
                self._cached_w_tilde[grid] = w_tilde.detach()

        # Fix broadcast dimensions:
        Nw, Nz = w_tilde.shape
        return w_tilde.view(Nw, *((1,) * (n_dim_tot - 2)), Nz)

    def get_w0(self) -> torch.Tensor:
        """Get G=0 component of first weight (rest all zero)."""
        if self.cache_w and (self._cached_w0 is not None):
            w0 = self._cached_w0
        else:
            Gzero = torch.zeros((1,), device=qp.rc.device)
            w0 = self.w(Gzero)[0] * self.G_sq_norm[0]  # others are zero at G=0
            if self.cache_w:
                self._cached_w0 = w0.detach()
        return w0

    def get_energy(self, n: qp.grid.FieldR) -> torch.Tensor:
        # Compute weighted densities:
        w_tilde = self.get_w_tilde(n.grid, len(n.data.shape) + 1)
        n_even, n_odd = n[None, ...].convolve(w_tilde).data.split(self.n_weights)
        n_odd_sq = (n_odd[:, None] * n_odd[None]).flatten(0, 1)[self.i_odd_pair]
        scalars = qp.grid.FieldR(n.grid, data=torch.cat((n_even, n_odd_sq), dim=0))

        # Evaluate free energy:
        f = qp.grid.FieldR(n.grid, data=self.f(scalars.data))
        return (scalars ^ f).sum(dim=0)

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        n_bar = torch.zeros((self.n_irred,) + n.shape, device=n.device, dtype=n.dtype)
        n_bar[0] = self.get_w0() * n
        return (n_bar * self.f(n_bar)).sum(dim=0)

    def bcast_parameters(self, comm: MPI.Comm) -> None:
        """Broadcast i.e. synchronize module parameters over `comm`."""
        if comm.size > 1:
            for parameter in self.parameters():
                comm.Bcast(qp.utils.BufferView(parameter.data))

    def allreduce_parameters_grad(self, comm: MPI.Comm) -> None:
        """Sum module parameter gradients over `comm`."""
        if comm.size > 1:
            for i_param, parameter in enumerate(self.parameters()):
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter.data)
                comm.Allreduce(MPI.IN_PLACE, qp.utils.BufferView(parameter.grad))


def get_pair_indices(N: int) -> torch.Tensor:
    """Get indices of distinct unordered pairs (i, j) for i, j in [0, N).
    The indices are to the flattened array of all pairs, i.e. in [0, N^2),
    (and will have length N(N+1)/2)."""
    index = torch.arange(N, dtype=torch.long, device=qp.rc.device)
    i, j = torch.where(index[:, None] <= index[None, :])
    return i * N + j

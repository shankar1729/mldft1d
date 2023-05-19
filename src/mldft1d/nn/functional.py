from __future__ import annotations
import qimpy as qp
import torch
import os
from mpi4py import MPI
from .basis import Basis, make_basis
from .function import Function, Linear


class Functional(torch.nn.Module):  # type: ignore
    """Machine-learned DFT in 1D."""

    n_weights: tuple[int, int]  #: Number of even and odd weight functions
    i_odd_pair: torch.Tensor  #: Indices of distinct pairs of odd weighted densities
    n_irred: int  #: Number of irreducible scalar combinations of weighted densities
    basis: Basis  #: Basis functions for expanding the nonlocal weight functions
    w: torch.nn.ModuleList  #: Linear combinations of even and odd basis
    f: Function  #: Per-particle free energy function of weighted densities
    use_local: bool  #: If true, local density/gradient count as an odd/even weight

    def __init__(
        self,
        comm: MPI.Comm,
        *,
        n_weights: tuple[int, int],
        basis: dict,
        f_hidden_sizes: list[int],
        use_local: bool = False,
    ) -> None:
        """Initializes functional with specified sizes (and random parameters)."""
        super().__init__()
        n_weights_even, n_weights_odd = n_weights
        self.n_weights = n_weights
        self.i_odd_pair = get_pair_indices(n_weights_odd)
        self.n_irred = n_weights_even + len(self.i_odd_pair)
        self.basis = make_basis(**basis)
        self.w = torch.nn.ModuleList(
            [
                Linear(
                    self.basis.n_basis,
                    n_weights_i - (1 if use_local else 0),
                    bias=False,
                    device=qp.rc.device,
                )
                for n_weights_i in n_weights
            ]
        )
        self.f = Function(self.n_irred, 1, f_hidden_sizes)
        self.use_local = use_local
        if comm.size > 1:
            self.bcast_parameters(comm)

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
            basis=self.basis.asdict(),
            f_hidden_sizes=self.f.n_hidden,
            use_local=self.use_local,
            state=self.state_dict(),
        )
        if comm.rank == 0:
            torch.save(params, filename)

    def get_w_tilde(self, grid: qp.grid.Grid, n_dim_tot: int = 2) -> torch.Tensor:
        """Compute weights for specified grid and total dimension count."""
        w_tildes = [w(basis) for w, basis in zip(self.w, self.basis(grid))]
        if self.use_local:
            w_tildes.insert(1, torch.ones_like(w_tildes[0][:1]))  # for local density
            w_tildes.append(grid.get_gradient_operator("H")[2:, 0, 0])  # for gradient
        w_tilde = torch.cat(w_tildes, dim=0)  # combine for efficient convolution below
        Nw, Nz = w_tilde.shape
        return w_tilde.view(Nw, *((1,) * (n_dim_tot - 2)), Nz)  # Add singleton dims

    def get_energy(self, n: qp.grid.FieldR) -> torch.Tensor:
        # Compute weighted densities:
        n_dim_tot = len(n.data.shape) + 1
        w_tilde = self.get_w_tilde(n.grid, n_dim_tot)
        n_even, n_odd = n[None, ...].convolve(w_tilde).data.split(self.n_weights)
        n_odd_sq = (n_odd[:, None] * n_odd[None]).flatten(0, 1)[self.i_odd_pair]
        scalars = qp.grid.FieldR(n.grid, data=torch.cat((n_even, n_odd_sq), dim=0))

        # Evaluate free energy:
        f = qp.grid.FieldR(n.grid, data=self.f(scalars.data)[0])
        return n ^ f

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        basis_o_even = self.basis.o[0]  # G=0 component of even basis (0 for odd)
        w_o_even = self.w[0](basis_o_even)
        n_bar = torch.zeros((self.n_irred,) + n.shape, device=n.device, dtype=n.dtype)
        n_bar[: len(w_o_even)] = torch.einsum("w, ... -> w...", w_o_even, n)
        if self.use_local:
            n_bar[len(w_o_even)] = n
        return n * self.f(n_bar)[0]

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

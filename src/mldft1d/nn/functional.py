from __future__ import annotations
import qimpy as qp
import numpy as np
import functools
import torch
import os
from mpi4py import MPI
from .function import Function


class Functional(torch.nn.Module):  # type: ignore
    """Machine-learned DFT in 1D."""

    n_weights: tuple[int, int]  #: Number of even and odd weight functions
    i_odd_pair: torch.Tensor  #: Indices of distinct pairs of odd weighted densities
    n_irred: int  #: Number of irreducible scalar combinations of weighted densities
    w: Function  #: Weight functions defining spatial nonlocality
    f: Function  #: Per-particle free energy function of weighted densities
    rc: float  #: Real space cutoff on weight functions
    cache_w: bool  #: Whether to precompute weight functions (must be off for training)
    _cached_w_tilde: dict[qp.grid.Grid, torch.Tensor]  #: cached weight functions

    def __init__(
        self,
        comm: MPI.Comm,
        *,
        n_weights: tuple[int, int],
        w_hidden_sizes: list[int],
        f_hidden_sizes: list[int],
        rc: float,
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
        self.rc = rc
        self.cache_w = cache_w
        if comm.size > 1:
            self.bcast_parameters(comm)
        self._cached_w_tilde = dict()

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
            rc=self.rc,
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
            w_tilde = get_weight_calculator(grid, self.rc)(self.w)

            # Make odd weights:
            w_tilde = w_tilde.to(torch.complex128)
            gradient_z = grid.get_gradient_operator("H")[2, 0]  # 1 x Nz array
            w_tilde[self.n_weights[0] :] *= gradient_z

            if self.cache_w:
                self._cached_w_tilde[grid] = w_tilde.detach()

        # Fix broadcast dimensions:
        Nw, Nz = w_tilde.shape
        return w_tilde.view(Nw, *((1,) * (n_dim_tot - 2)), Nz)

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
        n_bar[: self.n_weights[0]] = n
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


@functools.cache
def get_weight_calculator(grid: qp.grid.Grid, rc: float) -> WeightCalculatorR:
    return WeightCalculatorR(grid, rc)


class WeightCalculatorR:
    """Real-space weight calculator."""

    sup: int  #: supercell multiplier to accommodate range of kernel
    dz: float  #: grid spacing
    mine: slice  #: portion of recirpocal space to retain on present process
    z_sup_by_rc_sq: torch.Tensor  #: (z/rc)^2 in supercell (input to NN weight function)
    prefactor: torch.Tensor  #: weight function prefactor enforcing smooth falloff at rc

    def __init__(self, grid: qp.grid.Grid, rc: float) -> None:
        # Extract 1D grid properties:
        L = grid.lattice.Rbasis[2, 2].item()
        Nz = grid.shape[2]
        dz = L / Nz

        # Find supercell needed to accommodate rc:
        sup = int(np.ceil(2 * rc / L))
        Nz_sup = Nz * sup
        iz_sup = torch.arange(Nz_sup, device=qp.rc.device)[None]  # extra dim for bcast
        z_sup = dz * torch.where(iz_sup <= Nz_sup // 2, iz_sup, iz_sup - Nz_sup)

        # Store required quantities:
        self.sup = sup
        self.dz = dz
        self.mine = slice(grid.split2H.i_start, grid.split2H.i_stop)
        self.z_sup_by_rc_sq = torch.clamp((z_sup / rc).square(), max=1.0)
        self.prefactor = (1.0 - self.z_sup_by_rc_sq).square()  # C1 continuous at rc

    def __call__(self, w: Function) -> torch.Tensor:
        # Compute in real space (enforce positive, bounded and cutoff):
        w_real = self.prefactor / (1.0 + w(self.z_sup_by_rc_sq).square())

        # Convert to half reciprocal space and downselect to orginal cell:
        w_tilde = (torch.fft.rfft(w_real).real * self.dz)[:, :: self.sup]

        # Enforce G = 0 constraint:
        w_zero = w_tilde[..., :1]  # G = 0 component of w_tilde (with same dimensions)
        return w_tilde[..., self.mine] / w_zero

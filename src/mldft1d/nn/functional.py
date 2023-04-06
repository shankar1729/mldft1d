from __future__ import annotations
import numpy as np
import qimpy as qp
import torch
import os
import functools
from mpi4py import MPI
from typing import Protocol, Optional
from .function import Function


class Functional(torch.nn.Module):  # type: ignore
    """Machine-learned DFT in 1D."""

    n_weights: tuple[int, int]  #: Number of even and odd weight functions
    i_odd_pair: torch.Tensor  #: Indices of distinct pairs of odd weighted densities
    n_irred: int  #: Number of irreducible scalar combinations of weighted densities
    w: Function  #: Weight functions defining spatial nonlocality
    f: Function  #: Per-particle free energy function of weighted densities
    rc: float  #: If nonzero, compute weight function in real space with this cutoff
    Gc: float  #: If nonzero, smoothly cutoff reciprocal space w beyond Gc (in a0^-1)

    def __init__(
        self,
        comm: MPI.Comm,
        *,
        n_weights: tuple[int, int],
        w_hidden_sizes: list[int],
        f_hidden_sizes: list[int],
        rc: float = 0.0,
        Gc: float = 0.0,
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
        self.Gc = Gc
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
            w_hidden_sizes=self.w.n_hidden,
            f_hidden_sizes=self.f.n_hidden,
            rc=self.rc,
            Gc=self.Gc,
            state=self.state_dict(),
        )
        if comm.rank == 0:
            torch.save(params, filename)

    def get_w_tilde(self, grid: qp.grid.Grid, n_dim_tot: int) -> torch.Tensor:
        """Compute weights for specified grid and total dimension count."""
        calculator = get_weight_calculator(grid, self.rc, self.Gc)
        w_tilde = calculator(self.w, n_dim_tot).to(torch.complex128)
        n_weights_even, n_weights_odd = self.n_weights
        if n_weights_odd:
            # Multiply gradient operator to make a portion of the weights odd:
            gradient_z = grid.get_gradient_operator("H")[2]
            w_tilde[n_weights_even:] *= gradient_z
        return w_tilde

    def get_energy(self, n: qp.grid.FieldR) -> torch.Tensor:
        w_tilde = self.get_w_tilde(n.grid, len(n.data.shape) + 1)
        n_even, n_odd = n[None, ...].convolve(w_tilde).data.split(self.n_weights)
        n_odd_sq = (n_odd[:, None] * n_odd[None]).flatten(0, 1)[self.i_odd_pair]
        scalars = qp.grid.FieldR(n.grid, data=torch.cat((n_even, n_odd_sq), dim=0))
        f = qp.grid.FieldR(n.grid, data=self.f(scalars.data))
        return (scalars ^ f).sum(dim=0)

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        n_weights_even = self.n_weights[0]
        n_bar = torch.zeros((self.n_irred,) + n.shape, device=n.device, dtype=n.dtype)
        n_bar[:n_weights_even] = n[None]
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


class WeightCalculator(Protocol):
    def __call__(self, w: Function, n_dim_tot: int) -> torch.Tensor:
        ...


@functools.lru_cache()
def get_weight_calculator(grid: qp.grid.Grid, rc: float, Gc: float) -> WeightCalculator:
    if rc:
        return WeightCalculatorR(grid, rc)
    else:
        return WeightCalculatorG(grid, Gc)


class WeightCalculatorG:
    """Reciprocal-space weight calculator."""

    Gmag: torch.Tensor  #: reciprocal lattice vector magnitudes
    cutoff: Optional[torch.Tensor]  #: optional reciprocal space cutoff (if non-zero)

    def __init__(self, grid: qp.grid.Grid, Gc: float) -> None:
        iG = grid.get_mesh("H").to(torch.double)  # half-space
        self.Gmag = (iG @ grid.lattice.Gbasis.T).norm(dim=-1)
        self.cutoff = None
        if Gc:
            self.cutoff = torch.zeros_like(self.Gmag)
            sel = torch.where(self.Gmag < Gc)
            self.cutoff[sel] = ((self.Gmag[sel] * (np.pi / Gc)).cos() + 1.0) * 0.5

    def __call__(self, w: Function, n_dim_tot: int) -> torch.Tensor:
        bcast_shape = (1,) * (n_dim_tot - 3) + self.Gmag.shape
        w_tilde = w(self.Gmag.view(bcast_shape))
        w_zero = w_tilde[..., :1]  # G = 0 component of w_tilde (with same dimensions)
        result = w_tilde - (w_zero - 1.0)  # Constrain w(0) to 1
        if self.cutoff is not None:
            return result * self.cutoff.view(bcast_shape)
        else:
            return result


class WeightCalculatorR:
    """Real-space weight calculator."""

    sup: int  #: supercell multiplier to accommodate range of kernel
    dz: float  #: grid spacing
    z_sup_sq: torch.Tensor  #: z^2 in supercell (input to NN weight function)
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
        self.z_sup_sq = z_sup.square()
        self.prefactor = torch.where(
            abs(z_sup) < rc, 1.0 + torch.cos(z_sup * np.pi / rc), 0.0
        )  # value and derivative are both zero at rc

    def __call__(self, w: Function, n_dim_tot: int) -> torch.Tensor:
        # Compute in real space (enforce positive, bounded and cutoff):
        w_real = self.prefactor / (1.0 + w(self.z_sup_sq).square())
        # Convert to half reciprocal space and downselect to orginal cell:
        w_tilde = (torch.fft.rfft(w_real).real * self.dz)[:, :: self.sup]
        w_zero = w_tilde[..., :1]  # G = 0 component of w_tilde (with same dimensions)
        # Enforce G = 0 constraint and broadcast:
        Nw, Nz = w_tilde.shape
        bcast_shape = (Nw,) + (1,) * (n_dim_tot - 2) + (Nz,)
        return (w_tilde / w_zero).view(bcast_shape)


def get_pair_indices(N: int) -> torch.Tensor:
    """Get indices of distinct unordered pairs (i, j) for i, j in [0, N).
    The indices are to the flattened array of all pairs, i.e. in [0, N^2),
    (and will have length N(N+1)/2)."""
    index = torch.arange(N, dtype=torch.long, device=qp.rc.device)
    i, j = torch.where(index[:, None] <= index[None, :])
    return i * N + j

from __future__ import annotations
import numpy as np
import qimpy as qp
import torch
import os
import functools
from mpi4py import MPI
from typing import Protocol
from .function import Function
from .. import protocols


class Functional(torch.nn.Module):  # type: ignore
    """Machine-learned DFT in 1D."""

    w: Function  #: Weight functions defining spatial nonlocality
    f_scale: Function  #: Inhomogeneous scaling function of weighted densities
    f_bulk: protocols.FunctionBulk  #: Bulk free-energy function
    rc: float  #: If nonzero, compute weight function in real space with this cutoff

    def __init__(
        self,
        comm: MPI.Comm,
        *,
        n_weights: int,
        w_hidden_sizes: list[int],
        f_scale_hidden_sizes: list[int],
        f_bulk: protocols.FunctionBulk,
        rc: float = 0.0,
    ) -> None:
        """Initializes functional with specified sizes (and random parameters)."""
        super().__init__()
        self.w = Function(1, n_weights, w_hidden_sizes)
        self.f_scale = Function(n_weights, 1, f_scale_hidden_sizes)
        self.f_bulk = f_bulk
        self.rc = rc
        if comm.size > 1:
            self.bcast_parameters(comm)

    @classmethod
    def load(
        cls,
        comm: MPI.Comm,
        *,
        f_bulk: protocols.FunctionBulk,
        load_file: str = "",
        **kwargs,
    ) -> Functional:
        """
        Initialize functional from `kwargs` or from params file if given `load_file`.
        Any parameter in `kwargs` must be consistent with the params file, if given.
        The bulk free energy function `f_bulk` must always be specified explicitly.
        """
        params = dict(**kwargs)
        state = {}

        # Merge / check parameters from file, if available:
        if load_file and os.path.isfile(load_file):
            params_in = torch.load(load_file, map_location=qp.rc.device)
            for key, value in params_in.items():
                if key == "state":
                    state = value
                elif key == "f_bulk":
                    assert value == f_bulk.__class__.__name__
                elif key in params:
                    assert params[key] == value
                else:
                    params[key] = value

        # Create functional and load state if available:
        functional = Functional(comm, f_bulk=f_bulk, **params)
        if state:
            functional.load_state_dict(state)
        return functional

    def save(self, filename: str, comm: MPI.Comm) -> None:
        """Save parameters to specified filename."""
        params = dict(
            n_weights=self.w.n_out,
            w_hidden_sizes=self.w.n_hidden,
            f_scale_hidden_sizes=self.f_scale.n_hidden,
            f_bulk=self.f_bulk.__class__.__name__,
            rc=self.rc,
            state=self.state_dict(),
        )
        if comm.rank == 0:
            torch.save(params, filename)

    def get_w_tilde(self, grid: qp.grid.Grid, n_dim_tot: int) -> torch.Tensor:
        """Compute weights for specified grid and total dimension count."""
        return get_weight_calculator(grid, self.rc)(self.w, n_dim_tot)

    def get_energy(self, n: qp.grid.FieldR) -> torch.Tensor:
        w_tilde = self.get_w_tilde(n.grid, len(n.data.shape) + 1)
        n_bar = n[None, ...].convolve(w_tilde).data
        # TODO: odd and even weights
        n0 = n_bar[0]  # first weighted density alone,
        n0_rep = n0.expand(n_bar.shape)  # expanded to match dimension of all
        f = self.f_bulk.get_energy_bulk(n0)  # LDA-like (but with a weighted density)
        f *= (1.0 + self.f_scale(n_bar) - self.f_scale(n0_rep))[0]  # Enhancement factor
        return qp.grid.FieldR(n.grid, data=f).integral()

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        return self.f_bulk.get_energy_bulk(n)

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
def get_weight_calculator(grid: qp.grid.Grid, rc: float) -> WeightCalculator:
    if rc:
        return WeightCalculatorR(grid, rc)
    else:
        return WeightCalculatorG(grid)


class WeightCalculatorG:
    """Reciprocal-space weight calculator."""

    Gmag: torch.Tensor  #: reciprocal lattice vector magnitudes

    def __init__(self, grid: qp.grid.Grid) -> None:
        iG = grid.get_mesh("H").to(torch.double)  # half-space
        self.Gmag = (iG @ grid.lattice.Gbasis.T).norm(dim=-1)

    def __call__(self, w: Function, n_dim_tot: int) -> torch.Tensor:
        bcast_shape = (1,) * (n_dim_tot - 3) + self.Gmag.shape
        w_tilde = w(self.Gmag.view(bcast_shape))
        w_zero = w_tilde[..., :1]  # G = 0 component of w_tilde (with same dimensions)
        return w_tilde - (w_zero - 1.0)  # Return weight with w(0) constrained to 1


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

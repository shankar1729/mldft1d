from __future__ import annotations
import numpy as np
import qimpy as qp
import torch
import os
import functools
from mpi4py import MPI
from typing import TypeVar, Protocol
from hardrods1d.mlcdft.nn_function import NNFunction


NNInput = TypeVar("NNInput", torch.Tensor, qp.grid.FieldR)


class SchrodingerFunctional(torch.nn.Module):  # type: ignore
    """Machine-learned CDFT in 1D."""

    w: NNFunction  #: Weight functions defining spatial nonlocality
    f_ex: NNFunction  #: Free energy density as a function of weighted densities
    rc: float  #: If nonzero, compute weight function in real space with this cutoff

    def __init__(
        self,
        comm: MPI.Comm,
        *,
        n_weights: int,
        w_hidden_sizes: list[int],
        f_ex_hidden_sizes: list[int],
        rc: float = 0.0,
    ) -> None:
        """Initializes functional with specified sizes (and random parameters)."""
        super().__init__()
        self.T = 1.0 # Must be changed later
        self.w = NNFunction(1, n_weights, w_hidden_sizes)
        self.f_ex = NNFunction(n_weights, n_weights, f_ex_hidden_sizes)
        self.rc = rc
        if comm.size > 1:
            self.bcast_parameters(comm)

    @classmethod
    def load(
        cls,
        comm: MPI.Comm,
        *,
        load_file: str = "",
        **kwargs,
    ) -> SchrodingerFunctional:
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
        functional = SchrodingerFunctional(comm, **params)
        if state:
            functional.load_state_dict(state)
        return functional

    def save(self, filename: str, comm: MPI.Comm) -> None:
        """Save parameters to specified filename."""
        params = dict(
            n_weights=self.w.n_out,
            w_hidden_sizes=self.w.n_hidden,
            f_ex_hidden_sizes=self.f_ex.n_hidden,
            rc=self.rc,
            state=self.state_dict(),
        )
        if comm.rank == 0:
            torch.save(params, filename)

    def get_w_tilde(self, grid: qp.grid.Grid, n_dim_tot: int) -> torch.Tensor:
        """Compute weights for specified grid and total dimension count."""
        return get_weight_calculator(grid, self.rc)(self.w, n_dim_tot)

    def get_f_ex(self, n: NNInput) -> NNInput:
        if isinstance(n, torch.Tensor):
            n_zero = torch.zeros(
                (self.f_ex.n_in,) + (1,) * (len(n.shape) - 1), device=n.device
            )
            return self.f_ex(n)# - self.f_ex(n_zero)
        else:
            return qp.grid.FieldR(n.grid, data=self.get_f_ex(n.data))

    def get_energy(self, n: qp.grid.FieldR, V_minus_mu: qp.grid.FieldR) -> qp.Energy:
        w_tilde = self.get_w_tilde(n.grid, len(n.data.shape) + 1)
        n_bar = n[None, ...].convolve(w_tilde)
        energy = qp.Energy()
        energy["Omega0"] = n ^ (V_minus_mu)  # Ideal-gas part
        energy["Fex"] = (n_bar ^ self.get_f_ex(n_bar)).sum(dim=0)  # Excess
        return energy

    def get_energy_bulk(self, n_bulk: float, mu: float) -> torch.Tensor:
        """Compute bulk energy density."""
        n = torch.tensor(n_bulk, device=qp.rc.device)
        n_bar = n.repeat(self.w.n_out)
        return (n_bar @ self.get_f_ex(n_bar)) - mu * n

    def get_mu(self, n_bulk: float, create_graph: bool = False) -> torch.Tensor:
        """Compute chemical potential that will produce target density `n_bulk`."""
        # Bulk condition: (d/dn) [f_id(n) + f_ex(w * n)] = mu
        n = torch.tensor(n_bulk, device=qp.rc.device)
        n.requires_grad = True
        n_bar = n.repeat(self.w.n_out)
        energy_density = (n_bar @ self.get_f_ex(n_bar))
        return torch.autograd.grad(energy_density, n, create_graph=create_graph)[0]

    def bcast_parameters(self, comm: MPI.Comm) -> None:
        """Broadcast i.e. synchronize module parameters over `comm`."""
        if comm.size > 1:
            for parameter in self.parameters():
                comm.Bcast(qp.utils.BufferView(parameter.data))

    def allreduce_parameters_grad(self, comm: MPI.Comm) -> None:
        """Sum module parameter gradients over `comm`."""
        if comm.size > 1:
            for i_param, parameter in enumerate(self.parameters()):
                comm.Allreduce(MPI.IN_PLACE, qp.utils.BufferView(parameter.grad))


class WeightCalculator(Protocol):
    def __call__(self, w: NNFunction, n_dim_tot: int) -> torch.Tensor:
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

    def __call__(self, w: NNFunction, n_dim_tot: int) -> torch.Tensor:
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

    def __call__(self, w: NNFunction, n_dim_tot: int) -> torch.Tensor:
        # Compute in real space (enforce positive, bounded and cutoff):
        w_real = self.prefactor / (1.0 + w(self.z_sup_sq).square())
        # Convert to half reciprocal space and downselect to orginal cell:
        w_tilde = (torch.fft.rfft(w_real).real * self.dz)[:, :: self.sup]
        w_zero = w_tilde[..., :1]  # G = 0 component of w_tilde (with same dimensions)
        # Enforce G = 0 constraint and broadcast:
        Nw, Nz = w_tilde.shape
        bcast_shape = (Nw,) + (1,) * (n_dim_tot - 2) + (Nz,)
        return (w_tilde / w_zero).view(bcast_shape)

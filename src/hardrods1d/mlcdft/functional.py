from __future__ import annotations
import qimpy as qp
import torch
import os
from typing import TypeVar
from .nn_function import NNFunction


NNInput = TypeVar("NNInput", torch.Tensor, qp.grid.FieldR)


class Functional(torch.nn.Module):  # type: ignore
    """Machine-learned CDFT in 1D."""

    T: float  #: Temperature
    w: NNFunction  #: Weight functions defining spatial nonlocality
    f_ex: NNFunction  #: Free energy density as a function of weighted densities

    def __init__(
        self,
        *,
        T: float,
        w: NNFunction,
        f_ex: NNFunction,
    ) -> None:
        """Initializes to bulk fluid with no external potential."""
        super().__init__()
        self.T = T
        self.w = w
        self.f_ex = f_ex
        # Check dimensions:
        assert w.n_in == 1
        assert w.n_out == f_ex.n_in
        assert f_ex.n_out == w.n_out

    @classmethod
    def initialize(
        cls,
        *,
        T: float,
        n_weights: int,
        w_hidden_sizes: list[int],
        f_ex_hidden_sizes: list[int],
        load_params: str = "",
    ) -> Functional:
        """
        Initialize functional from specified sizes or from params file if `load_params`.
        """
        functional = Functional(
            T=T,
            w=NNFunction(1, n_weights, w_hidden_sizes),
            f_ex=NNFunction(n_weights, n_weights, f_ex_hidden_sizes),
        )
        if load_params and os.path.isfile(load_params):
            params = torch.load(load_params, map_location=qp.rc.device)
            functional.load_state_dict(params["state"])
        return functional

    def save(self, filename: str) -> None:
        """Save parameters to specified filename."""
        params = dict(
            T=self.T,
            n_weights=self.w.n_out,
            w_hidden_sizes=self.w.n_hidden,
            f_ex_hidden_sizes=self.f_ex.n_hidden,
            state=self.state_dict(),
        )
        torch.save(params, filename)

    def get_w_tilde(self, grid: qp.grid.Grid, n_dim_tot: int) -> torch.Tensor:
        """Compute weights for specified grid and total dimension count."""
        # Compute G = 0 correction:
        Gzero = torch.zeros((1,) * n_dim_tot, device=qp.rc.device)
        w_zero = self.w(Gzero)
        w_zero -= 1.0
        # Compute G-dependent result:
        Gmag_shape = (1,) * (n_dim_tot - 3) + grid.shapeH
        Gmag = Functional.Gmag(grid).view(Gmag_shape)
        return self.w(Gmag) - w_zero

    def get_f_ex(self, n: NNInput) -> NNInput:
        if isinstance(n, torch.Tensor):
            n_zero = torch.zeros(
                (self.f_ex.n_in,) + (1,) * (len(n.shape) - 1), device=n.device
            )
            return self.f_ex(n) - self.f_ex(n_zero)
        else:
            return qp.grid.FieldR(n.grid, data=self.get_f_ex(n.data))

    def get_energy(self, n: qp.grid.FieldR, V_minus_mu: qp.grid.FieldR) -> qp.Energy:
        w_tilde = self.get_w_tilde(n.grid, len(n.data.shape) + 1)
        n_bar = n[None, ...].convolve(w_tilde)
        energy = qp.Energy()
        energy["Omega0"] = n ^ (self.T * n.log() + V_minus_mu)  # Ideal-gas part
        energy["Fex"] = self.T * (n_bar ^ self.get_f_ex(n_bar)).sum(dim=0)  # Excess
        return energy

    def get_mu(self, n_bulk: float, create_graph: bool = False) -> torch.Tensor:
        """Compute chemical potential that will produce target density `n_bulk`."""
        # Bulk condition: (d/dn) [f_id(n) + f_ex(w * n)] = mu
        n = torch.tensor(n_bulk, device=qp.rc.device)
        n.requires_grad = True
        n_bar = n.repeat(self.w.n_out)
        energy_density = self.T * (n * n.log() + n_bar @ self.get_f_ex(n_bar))
        return torch.autograd.grad(energy_density, n, create_graph=create_graph)[0]

    @staticmethod
    def Gmag(grid: qp.grid.Grid):
        iG = grid.get_mesh("H").to(torch.double)  # half-space
        return (iG @ grid.lattice.Gbasis.T).norm(dim=-1)

import qimpy as qp
import torch
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
            return self.f_ex(n)
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
        w_bulk = self.w(torch.zeros(1, device=qp.rc.device))
        n = torch.tensor(n_bulk)
        n.requires_grad = True
        n_bar = w_bulk * n
        energy_density = self.T * (n * n.log() + n_bar @ self.get_f_ex(n_bar))
        return torch.autograd.grad(energy_density, n, create_graph=create_graph)[0]

    @staticmethod
    def Gmag(grid: qp.grid.Grid):
        iG = grid.get_mesh("H").to(torch.double)  # half-space
        return (iG @ grid.lattice.Gbasis.T).norm(dim=-1)

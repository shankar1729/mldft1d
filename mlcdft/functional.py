import qimpy as qp
import numpy as np
import torch
from .nn_function import NNFunction


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

    def get_energy(self, n: qp.grid.FieldR, V_minus_mu: qp.grid.FieldR) -> qp.Energy:
        w_tilde = self.w(Functional.Gmag(n.grid)[None])
        n_bar = n[None, ...].convolve(w_tilde)

        energy = qp.Energy()
        energy["Omega0"] = n ^ (self.T * n.log() + V_minus_mu)  # Ideal-gas part
        energy["Fex"] = self.T * (n_bar ^ self.f_ex(n_bar)).sum()  # Excess part
        return energy

    def get_mu(self, n_bulk: float) -> float:
        """Compute chemical potential that will produce target density `n_bulk`."""
        # Bulk condition: (d/dn) [f_id(n) + f_ex(w * n)] = mu
        w_bulk = self.w(torch.zeros(1, device=qp.rc.device))
        n = torch.tensor(n_bulk)
        n.requires_grad = True
        n_bar = w_bulk * n
        (self.T * (n * n.log() + n_bar @ self.f_ex(n_bar))).backward()
        return n.grad.item()

    @staticmethod
    def Gmag(grid: qp.grid.Grid):
        iG = grid.get_mesh("H").to(torch.double)  # half-space
        return (iG @ grid.lattice.Gbasis.T).norm(dim=-1)

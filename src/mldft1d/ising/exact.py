import qimpy as qp
import numpy as np
from .. import Grid1D
from scipy.optimize import fsolve
from typing import Optional
import torch


class Exact:
    """Exact 1D Ising model solver."""

    grid1d: Grid1D
    n_bulk: float  #: Bulk number density of the fluid
    T: float  #: Fermi smearing width
    J: float  #: Exchange interaction
    mu: float  #: Bulk chemical potential
    n: qp.grid.FieldR  #: Equilibrium density
    V: qp.grid.FieldR  #: External potential
    energy: qp.Energy  #: Equilibrium energy components

    def __init__(self, grid1d: Grid1D, *, n_bulk: float, T: float, J: float) -> None:
        """Initializes to bulk fluid with no external potential."""
        self.grid1d = grid1d
        self.n_bulk = n_bulk
        self.T = T
        self.J = J
        self.mu = -self.bulk_potential(torch.tensor(n_bulk, device=qp.rc.device)).item()
        self.n = qp.grid.FieldR(
            grid1d.grid,
            data=torch.full(grid1d.grid.shapeR_mine, n_bulk, device=qp.rc.device),
        )
        self.V = self.n.zeros_like()
        self.energy = qp.Energy()

    def known_part(self) -> Optional[tuple[float, qp.grid.FieldR]]:
        """No known part: learn the entire part in ML."""
        return None

    def minimize(self) -> qp.Energy:
        # Compute density:
        n = self.n.data[0, 0]
        n[:] = self.unmap(fsolve(self.root_function, self.map(n)))

        # Compute energy by numerically integrating exact potential = -dEint/dn:
        energy = self.energy
        n_ref = 0.0  # define unpolarized system to have zero energy
        n_steps = 100  # number of integration steps
        dn = (n - n_ref) / n_steps
        i_step = torch.arange(n_steps, device=qp.rc.device) + 0.5  # interval midpoints
        n_path = n_ref + torch.outer(i_step, dn)
        V_path = self.exact_potential(n_path)
        energy["Int"] = (V_path @ dn).sum().item() * (-self.grid1d.dz)  # internal
        energy["Ext"] = ((self.V - self.mu) ^ self.n).sum().item()
        return energy

    def F(self, n: torch.Tensor, n_prime: torch.Tensor) -> torch.Tensor:
        """Function defined below Eq. 22 of Percus ref."""
        e = self.e
        f = e - 1.0
        return 0.5 * ((1.0 + e) * n - f * n_prime) + torch.sqrt(
            e + 0.25 * f * (f * (n - n_prime).square() - 4 * n * n_prime)
        )

    def bulk_potential(self, n: torch.Tensor) -> torch.Tensor:
        """Solution of potential from Eq. 22 for a spatially uniform n."""
        return self.J - self.T * torch.log(
            self.F(n, n) ** 2 / (self.e**2 * (1.0 - n.square()))
        )

    def exact_potential(self, n: torch.Tensor) -> torch.Tensor:
        """Solution of potential from Eq. 22 for inhomogeneous n."""
        return self.J - self.T * torch.log(
            self.F(n, torch.roll(n, +1, dims=-1))
            * self.F(n, torch.roll(n, -1, dims=-1))
            / (self.e**2 * (1.0 - n.square()))
        )

    @property
    def e(self) -> float:
        """e parameter defined in Eq. 16 of Percus ref."""
        return np.exp(-self.J / self.T)

    @staticmethod
    def map(n: torch.Tensor) -> np.ndarray:
        """Map n on (-1, 1) to independent variable on (-infty, infty)."""
        return torch.atanh(n).to(qp.rc.cpu).numpy()

    @staticmethod
    def unmap(n_mapped: np.ndarray) -> torch.Tensor:
        """Inverse of `map`."""
        return torch.tanh(torch.from_numpy(n_mapped).to(qp.rc.device))

    def root_function(self, n_mapped: np.ndarray) -> np.ndarray:
        """Eq. 22 of Percus ref. cast as a root function to solve for n."""
        n = Exact.unmap(n_mapped)
        V = self.V.data[0, 0]
        V_error = self.exact_potential(n) - (V - self.mu)
        return V_error.to(qp.rc.cpu).numpy()

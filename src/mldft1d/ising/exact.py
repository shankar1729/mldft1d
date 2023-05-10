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

    def known_V(self) -> Optional[qp.grid.FieldR]:
        """No known potential: learn the entire part in ML."""
        return None

    def minimize(self) -> qp.Energy:
        # Compute density and energy
        energy = self.energy
        n_data = self.n.data[0, 0]
        self.n.data[0, 0] = self.unmap(fsolve(self.root_function, self.map(n_data)))
        return energy  # TODO: actually compute energy

    def E(self, n: torch.Tensor, n_prime: torch.Tensor) -> torch.Tensor:
        """Function defined below Eq. 20 of Percus ref."""
        e = self.e
        f = e - 1.0
        return (
            (1.0 + e) * n
            + (1.0 - e) * n_prime
            - 1.0
            + torch.sqrt((1.0 + f * (n + n_prime)) ** 2 - 4.0 * e * f * n * n_prime)
        )

    def bulk_potential(self, n: torch.Tensor) -> torch.Tensor:
        """Solution of U from Eq. 19 for a spatially uniform rho."""
        return -self.T * torch.log(
            self.E(n, n) ** 2 / (4.0 * self.e**2 * n * (1.0 - n))
        )

    @property
    def e(self) -> float:
        """e parameter defined in Eq. 16 of Percus ref."""
        return np.exp(-self.J / self.T)

    @staticmethod
    def map(n: torch.Tensor) -> np.ndarray:
        """Map n on (0, 1) to independent variable on (-infty, infty)."""
        return torch.special.logit(n).to(qp.rc.cpu).numpy()

    @staticmethod
    def unmap(n_mapped: np.ndarray) -> torch.Tensor:
        """Inverse of `map`."""
        return torch.special.expit(torch.from_numpy(n_mapped).to(qp.rc.device))

    def root_function(self, n_mapped: np.ndarray) -> np.ndarray:
        """Eq. 20 of Percus ref. cast as a root function to solvqe for n."""
        n = Exact.unmap(n_mapped)
        V = self.V.data[0, 0]
        V_error = (
            V
            - self.mu
            + self.T
            * torch.log(
                self.E(n, torch.roll(n, +1))
                * self.E(n, torch.roll(n, -1))
                / (4.0 * self.e**2 * n * (1.0 - n))
            )
        )
        return V_error.to(qp.rc.cpu).numpy()

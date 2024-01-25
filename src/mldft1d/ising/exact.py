from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve
import torch

from qimpy import rc, Energy
from qimpy.grid import FieldR
from .. import Grid1D
from ..protocols import get_mu
from .ideal_spin_gas import IdealSpinGas


class Exact:
    """Exact 1D Ising model solver."""

    grid1d: Grid1D
    n_bulk: torch.Tensor  #: Bulk number density of the fluid
    T: float  #: Fermi smearing width
    J: float  #: Exchange interaction
    mu: torch.Tensor  #: Bulk chemical potential
    n: FieldR  #: Equilibrium density
    V: FieldR  #: External potential
    energy: Energy  #: Equilibrium energy components

    def __init__(
        self, grid1d: Grid1D, *, n_bulk: torch.Tensor, T: float, J: float
    ) -> None:
        """Initializes to bulk fluid with no external potential."""
        self.grid1d = grid1d
        self.n_bulk = n_bulk
        self.T = T
        self.J = J
        self.mu = get_mu([IdealSpinGas(T), BulkExcess(T, J)], n_bulk)
        self.n = FieldR(
            grid1d.grid,
            data=n_bulk.view(-1, 1, 1, 1).repeat((1,) + grid1d.grid.shapeR_mine),
        )
        self.V = self.n.zeros_like()
        self.energy = Energy()

    def training_targets(self) -> Optional[tuple[float, FieldR]]:
        """Return ideal gas as known part."""
        n = self.n.clone()
        n.data.requires_grad = True
        n.data.grad = None
        Eideal = IdealSpinGas(self.T).get_energy(n)
        (Eideal / n.grid.dV).backward()  # functional derivative -> n.data.grad
        Videal = FieldR(n.grid, data=n.data.grad)
        V_minus_mu = FieldR(self.V.grid, data=(self.V.data - self.mu.view(-1, 1, 1, 1)))
        return (self.energy["Int"] - Eideal).item(), -(Videal + V_minus_mu)

    def minimize(self) -> Energy:
        # Compute density:
        n = self.n.data[0, 0, 0]
        n[:] = self.unmap(fsolve(self.root_function, self.map(n)))

        # Compute energy by numerically integrating exact potential = -dEint/dn:
        energy = self.energy
        n_steps = 100  # number of integration steps
        n_ref = n.mean()[None]
        E_ref = self.grid1d.L * (
            IdealSpinGas(self.T).get_energy_bulk(n_ref).item()
            + BulkExcess(self.T, self.J).get_energy_bulk(n_ref).item()
        )
        dn = (n - n_ref) / n_steps
        i_step = torch.arange(n_steps, device=rc.device) + 0.5  # interval midpoints
        n_path = n_ref + torch.outer(i_step, dn)
        V_path = self.exact_potential(n_path)
        V_minus_mu = FieldR(self.V.grid, data=(self.V.data - self.mu.view(-1, 1, 1, 1)))
        energy["Int"] = E_ref + (V_path @ dn).sum().item() * (-self.grid1d.dz)
        energy["Ext"] = (V_minus_mu ^ self.n).sum().item()
        return energy

    def E(self, n: torch.Tensor, n_prime: torch.Tensor) -> torch.Tensor:
        """Function defined below Eq. 20 of Percus ref."""
        e = self.e
        f = e - 1
        return ((1 + e) * n - f * n_prime - 1) + torch.sqrt(
            (1 + f * (n + n_prime)) ** 2 - 4 * e * f * n * n_prime
        )

    def exact_potential(self, n: torch.Tensor) -> torch.Tensor:
        """Solution of potential from Eq. 20."""
        return self.J - self.T * torch.log(
            self.E(n, torch.roll(n, +1, dims=-1))
            * self.E(n, torch.roll(n, -1, dims=-1))
            / (4 * self.e**2 * n * (1 - n))
        )

    @property
    def e(self) -> float:
        """e parameter defined in Eq. 16 of Percus ref."""
        return np.exp(-self.J / self.T)

    @staticmethod
    def map(n: torch.Tensor) -> np.ndarray:
        """Map n on (0, 1) to independent variable on (-infty, infty)."""
        return torch.special.logit(n).to(rc.cpu).numpy()

    @staticmethod
    def unmap(n_mapped: np.ndarray) -> torch.Tensor:
        """Inverse of `map`."""
        return torch.special.expit(torch.from_numpy(n_mapped).to(rc.device))

    def root_function(self, n_mapped: np.ndarray) -> np.ndarray:
        """Eq. 22 of Percus ref. cast as a root function to solve for n."""
        n = Exact.unmap(n_mapped)
        V = self.V.data[0, 0, 0]
        V_error = self.exact_potential(n) - (V - self.mu)
        return V_error.to(rc.cpu).numpy()


@dataclass
class BulkExcess:
    """Exact excess free energy of homogeneous Ising model in 1D."""

    T: float
    J: float

    def get_energy_bulk(self, n_in: torch.Tensor) -> torch.Tensor:
        n = n_in[..., 0]
        e = np.exp(-self.J / self.T)
        f = e - 1
        E = 2 * n - 1 + torch.sqrt(1 + 4 * f * n * (1 - n))
        return self.J * n + self.T * (
            n * torch.log(E.square() / (4 * n.square()))
            - torch.log(1 - n + E / (2 * e))
        )

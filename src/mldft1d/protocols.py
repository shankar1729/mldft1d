from __future__ import annotations
from typing import Protocol, Sequence
import qimpy as qp
import torch


class Functional(Protocol):
    """A term in electronic/classical DFT, such as an excess functional."""

    def get_energy(self, n: qp.grid.FieldR) -> torch.Tensor:
        """Compute term in energy of inhomogeneous system with density `n`.
        Returns energy as a tensor to support autograd."""

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        """Compute term in energy density of homogeneous system with bulk density `n`.
        Inputs and outputs are tensors to support autograd."""


class Variational(Protocol):
    """An overall DFT that minimizes energy for a given external potential."""

    V: qp.grid.FieldR  #: External potential
    energy: qp.Energy  #: Equilibrium energy components, available after `minimize`

    def minimize(self) -> qp.Energy:
        """Solve Euler-Lagrange equation and return equilibrium energy."""

    @property
    def n(self) -> qp.grid.FieldR:
        """Equilibrium density, which should be available after `minimize`."""


def get_mu(functionals: Sequence[Functional], n_bulk: float) -> float:
    """Return chemical potential corresponding to bulk density `n_bulk`."""
    energy_bulk = torch.tensor(0.0, device=qp.rc.device)
    n = torch.tensor(n_bulk, device=qp.rc.device)
    n.requires_grad = True
    for functional in functionals:
        energy_bulk += functional.get_energy_bulk(n)
    return torch.autograd.grad(energy_bulk, n)[0].item()

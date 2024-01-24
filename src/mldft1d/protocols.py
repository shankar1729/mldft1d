from __future__ import annotations
from typing import Protocol, Sequence, Optional

import torch

from qimpy import Energy
from qimpy.grid import FieldR


class FunctionBulk(Protocol):
    """A term in the bulk (free-)energy function of electronic/classical DFT."""

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        """Compute term in energy density of homogeneous system with bulk density `n`.
        Inputs and outputs are tensors to support autograd and batching.
        Dimensions are [n_batch x] n_sites at input, and [n_batch] for output,
        where n_sites are the number of fluid sites / electron spins etc."""


class Functional(FunctionBulk, Protocol):
    """A term in electronic/classical DFT, such as an excess functional."""

    def get_energy(self, n: FieldR) -> torch.Tensor:
        """Compute term in energy of inhomogeneous system with density `n`.
        Returns energy as a tensor to support autograd and batching.
        Input has dimensions [n_batch x] n_sites x grid-dimensions,
        while output has dimensions [n_batch]."""


class DFT(Protocol):
    """An overall DFT that minimizes energy for a given external potential."""

    mu: torch.Tensor  #: Chemical potential on each site (length n_sites)
    V: FieldR  #: External potential(s), n_sites x grid_dimensions
    n: FieldR  #: Equlibrium density(ies), available after `minimize`
    energy: Energy  #: Equilibrium energy components, available after `minimize`

    def minimize(self) -> Energy:
        """Solve Euler-Lagrange equation and return equilibrium energy."""

    def training_targets(self) -> Optional[tuple[float, FieldR]]:
        """Target part of equilibrium energy and potential for ML potentials, if any."""


def get_mu(functionals: Sequence[FunctionBulk], n_bulk: torch.Tensor) -> torch.Tensor:
    """Return chemical potentials corresponding to bulk densities `n_bulk`."""
    energy_bulk = torch.tensor(0.0, device=n_bulk.device)
    n = n_bulk.detach().clone()
    n.requires_grad = True
    for functional in functionals:
        energy_bulk += functional.get_energy_bulk(n)
    return torch.autograd.grad(energy_bulk, n)[0].detach()

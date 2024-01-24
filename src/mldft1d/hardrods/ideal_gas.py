import qimpy as qp
import torch


class IdealGas:
    """Ideal gas kinetic energy and entropy."""

    T: float  #: temperature

    def __init__(self, T: float):
        self.T = T

    def get_energy(self, n: qp.grid.FieldR) -> torch.Tensor:
        return self.T * (n ^ n.log()).sum(dim=-1)

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        return self.T * (n * n.log()).sum(dim=-1)

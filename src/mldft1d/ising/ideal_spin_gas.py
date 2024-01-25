import qimpy as qp
import torch


class IdealSpinGas:
    """Ideal spin gas kinetic energy and entropy."""

    T: float  #: temperature

    def __init__(self, T: float):
        self.T = T

    def get_energy(self, n: qp.grid.FieldR) -> torch.Tensor:
        n_conj = 1.0 - n
        return self.T * ((n ^ n.log()) + (n_conj ^ n_conj.log())).sum(dim=-1)

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        n_conj = 1.0 - n
        return self.T * (n * n.log() + n_conj * n_conj.log()).sum(dim=-1)

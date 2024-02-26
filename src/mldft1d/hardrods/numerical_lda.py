import torch
import numpy as np

from qimpy import rc
from qimpy.grid import FieldR


class NumericalLDA:
    """Ideal gas kinetic energy and entropy."""

    dn_inv: float  #: inverse spacing in density in look-up table
    n_max: float  #: maximum value in look-up table
    a: torch.Tensor  #: values in look-up table
    a_prime: torch.Tensor  #: derivatives in look-up table

    def __init__(self, filename: str):
        n, a, a_n = np.loadtxt(filename).T
        dn = n[1] - n[0]
        self.dn_inv = 1 / dn
        self.n_max = n[-1] - 1e-6 * dn  # margin on upper end
        self.a = torch.from_numpy(a).to(rc.device)
        self.a_prime = torch.from_numpy(a_n * dn).to(rc.device)

    def calculate(self, n: torch.Tensor) -> torch.Tensor:
        x = self.dn_inv * n.clamp(min=0.0, max=self.n_max)
        i = torch.floor(x).detach().to(torch.long)
        t = x - i
        t_bar = 1.0 - t
        # Cubic bezier spline using values and derivatives in interval
        a0 = self.a[i]
        a3 = self.a[i + 1]
        a1 = a0 + self.a_prime[i] / 3.0
        a2 = a3 - self.a_prime[i + 1] / 3.0
        result = torch.stack((a0, a1, a2, a3), dim=0)
        while len(result) > 1:
            result = t_bar * result[:-1] + t * result[1:]  # de Casteljau algorithm
        return result[0]

    def get_energy(self, n: FieldR) -> torch.Tensor:
        return FieldR(n.grid, data=self.calculate(n.data[0])).integral()

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        return self.calculate(n[..., 0])

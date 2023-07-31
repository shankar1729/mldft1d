from dataclasses import dataclass, field

from scipy.special import roots_legendre
import numpy as np
import torch

from qimpy import rc
from qimpy.grid import FieldR


@dataclass
class BulkExchange:
    """Bulk exchange-energy density for 1D soft-Coulomb interaction."""

    a: float = 1.0

    # Gauss integration quadrature over 0 to 2:
    quad_nodes: torch.Tensor = field(init=False)  #: quadrature nodes
    quad_weights: torch.Tensor = field(init=False)  #: quadrature weights
    integrand0: torch.Tensor = field(init=False)  #: constant singular part of integrand
    singular_part: float = field(init=False)  #: precomputed singular-part of integral

    def __post_init__(self):
        """Initialize quadrature weights"""
        roots, weights = roots_legendre(15)
        roots += 1.0  # switch to [0, 2] integration domain
        weights *= 2.0 - roots  # include common (2-x) factor in integrand here
        self.quad_nodes = torch.from_numpy(roots).to(rc.device)
        self.quad_weights = torch.from_numpy(weights).to(rc.device)
        self.integrand0 = torch.log(self.quad_nodes)
        self.singular_part = 3 - np.log(4)

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        kFa = (0.5 * np.pi * self.a) * n
        integrand = torch.special.modified_bessel_k0(kFa[..., None] * self.quad_nodes)
        integral = (integrand + self.integrand0) @ self.quad_weights
        return -(0.5 * n).square() * (integral + self.singular_part)

    def get_energy(self, n: FieldR) -> torch.Tensor:
        """Energy of inhomogeneous system in a local-density approximation."""
        energy_density = FieldR(n.grid, data=self.get_energy_bulk(n.data))
        return energy_density.integral()


if __name__ == "__main__":

    def main():
        import matplotlib.pyplot as plt
        import pylibxc

        n = np.logspace(-3, +1, 200)
        func = pylibxc.LibXCFunctional("lda_x_1d_soft", 1)
        eps_x_libxc = func.compute({"rho": n})["zk"][:, 0]
        eps_x = BulkExchange().get_energy_bulk(torch.from_numpy(n)).numpy() / n
        plt.plot(n, -eps_x_libxc, label="LibXC")
        plt.plot(n, -eps_x, label="Internal")
        plt.plot(n, np.abs(eps_x_libxc - eps_x), label="Error")
        plt.xlabel("$n$")
        plt.ylabel(r"Per-particle exchaneg $-\epsilon_X$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()

    main()

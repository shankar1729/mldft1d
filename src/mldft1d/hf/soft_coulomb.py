from dataclasses import dataclass

import torch
import numpy as np

from qimpy import rc


@dataclass
class SoftCoulomb:
    """1D Soft-Coulomb interaction in real and reciprocal space."""

    a: float = 1.0  # Softening length-scale

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate in real-space."""
        return 1.0 / torch.sqrt(self.a**2 + r.square())

    def tilde(self, G: torch.Tensor) -> torch.Tensor:
        """Evaluate in reciprocal space."""
        return 2.0 * torch.special.modified_bessel_k0(torch.abs(G * self.a))

    def periodic_kernel(self, G: torch.Tensor) -> torch.Tensor:
        """Same as tilde, but with G=0 component projected out."""
        result = self.tilde(G)
        result[G == 0.0] = 0.0
        return result

    def truncated_kernel(self, Nz: int, dz: float, real: bool) -> torch.Tensor:
        # Evaluate on a fine enough real-space grid
        scale = np.ceil(3 * dz / self.a)
        dz_fine = dz / scale
        Nz_fine = Nz * scale
        z_fine = dz_fine * torch.cat(
            (
                torch.arange(Nz_fine // 2, device=rc.device),
                torch.arange(Nz_fine // 2 - Nz_fine, 0, device=rc.device),
            )
        )
        K_fine = self(z_fine)
        # Fourier transform and down-sample to orginal
        K_fine_tilde = torch.fft.fft(K_fine).real * dz_fine
        Nz_half = Nz // 2
        K_tilde_pos = K_fine_tilde[: (Nz_half + 1)]
        if real:
            return K_tilde_pos
        else:
            return torch.cat((K_tilde_pos, K_fine_tilde[(Nz_half + 1 - Nz) :]))


if __name__ == "__main__":

    def main():
        import matplotlib.pyplot as plt
        from qimpy.grid import FieldR
        from .. import Grid1D, get1D

        grid1d = Grid1D(L=30.0, dz=0.1)

        # Create neutral charge distribution to test"
        rho = FieldR(grid1d.grid)
        iz0 = 10
        z0 = grid1d.dz * iz0
        rho.data[0, 0, 0] = 1.0 / grid1d.dz
        rho.data[0, 0, iz0] = -1.0 / grid1d.dz

        sc = SoftCoulomb()
        kernel = sc.tilde(grid1d.Gmag)
        kernel[0, 0, 0] = 0.0  # project out G=0
        numeric = rho.convolve(kernel).data
        analytic = torch.zeros_like(numeric)
        for sign, location in ((+1, 0.0), (-1, z0)):
            for i_lattice in range(-11, 13):
                analytic += sign * sc(grid1d.z - (i_lattice * grid1d.L + location))
        analytic -= analytic.mean()

        z = get1D(grid1d.z)
        plt.plot(z, get1D(numeric), label="numeric")
        plt.plot(z, get1D(analytic), label="analytic")
        plt.plot(z, get1D(numeric - analytic), label="difference")
        plt.axhline(0, color="k", ls="dotted", lw=1)
        plt.legend()
        plt.show()

    main()

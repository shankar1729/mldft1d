import torch
import qimpy as qp
import numpy as np
from dataclasses import dataclass
from scipy.special import hermite, loggamma
from abc import abstractmethod


@dataclass
class Basis:
    """Abstract base class for weight function basis sets."""

    n_basis: int  #: total number of basis functions

    @abstractmethod
    def get(self, G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get even and odd basis functions for specified reciprocal lattice.
        Each output should be sized (n_basis_odd/even,) + G.shape."""
        ...


@dataclass
class Hermite(Basis):
    """Eigenbasis of quantum Harmonic oscillator."""

    sigma: float  #: Gaussian width of ground state

    def get(self, G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute Hermite polynomials (using scipy):
        sigmaG = self.sigma * G.to(qp.rc.cpu).numpy()
        i_bases = np.arange(self.n_basis)
        basis_np = np.array(
            [hermite(i_basis, monic=True)(sigmaG) for i_basis in i_bases]
        )

        # Multiply Gaussian and normalization factors:
        basis_np *= np.exp(-0.5 * np.square(sigmaG))[None]
        prefactor = np.exp(
            0.5 * ((np.log(2) + 1j * np.pi) * i_bases - loggamma(i_bases + 1))
        ) * np.sqrt(self.sigma * 2 * np.sqrt(np.pi))
        basis = torch.from_numpy(np.einsum("b, b... -> b...", prefactor, basis_np)).to(
            G.device
        )
        return basis[::2], basis[1::2]  # Spit even and odd basis


def main():
    """Plot basis functions for testing."""
    from ..grid1d import Grid1D, get1D
    import matplotlib.pyplot as plt

    L = 15.0
    dz = 0.05
    grid1d = Grid1D(L=L, dz=dz)
    G = grid1d.iGz * (2 * np.pi / L)
    basis = Hermite(n_basis=51, sigma=0.5)
    z_np = get1D(grid1d.z)
    for label, style, w_tilde in zip(("Even", "Odd"), ("r", "b"), basis.get(G)):
        w = torch.fft.irfft(w_tilde.detach()[:, 0, 0]).real / dz
        plt.plot(z_np, w.to(qp.rc.cpu).numpy().T, style, label=f"{label} weights")
        overlap = (w @ w.T) * dz
        print(f"\nOverlap matrix for {label} weights:\n{qp.utils.fmt(overlap)}")
    plt.xlim(0, 0.5 * L)
    plt.xlabel("$z$")
    plt.ylabel("$w(z)$")
    plt.show()


if __name__ == "__main__":
    main()

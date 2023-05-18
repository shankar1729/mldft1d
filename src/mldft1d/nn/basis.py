import torch
import qimpy as qp
import numpy as np
from dataclasses import dataclass, asdict
from scipy.special import hermite, loggamma
from functools import cache, cached_property
from abc import abstractmethod


@dataclass(eq=True, frozen=True)
class Basis:
    """Abstract base class for weight function basis sets."""

    n_basis: int  #: total number of basis functions
    r_max: float  #: nominal spatial extent of basis (need not be a sharp cutoff)

    @abstractmethod
    def get(self, G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get even and odd basis functions for specified reciprocal lattice.
        Each output should be sized (n_basis_odd/even,) + G.shape."""
        ...

    @cache
    def __call__(self, grid: qp.grid.Grid) -> tuple[torch.Tensor, torch.Tensor]:
        """Interface to calculate basis for a (1D) grid."""
        Gz = grid.get_gradient_operator("H")[2, 0, 0].imag
        return self.get(Gz)

    @cached_property
    def o(self) -> tuple[torch.Tensor, torch.Tensor]:
        """G=0 component of basis functions."""
        Gzero = torch.tensor(0.0, device=qp.rc.device)
        return self.get(Gzero)

    def asdict(self) -> dict:
        result = asdict(self)
        result["type"] = self.__class__.__name__.lower()
        return result


class Hermite(Basis):
    """Eigenbasis of quantum Harmonic oscillator."""

    @property
    def sigma(self) -> float:
        """Gaussian width of ground state."""
        return self.r_max / np.sqrt(2 * self.n_basis)

    def get(self, G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute Hermite polynomials (using scipy):
        sigma = self.sigma
        sigmaG = sigma * G.to(qp.rc.cpu).numpy()
        i_bases = np.arange(self.n_basis)
        basis_np = np.array(
            [hermite(i_basis, monic=True)(sigmaG) for i_basis in i_bases]
        )

        # Multiply Gaussian and normalization factors:
        basis_np *= np.exp(-0.5 * np.square(sigmaG))[None]
        prefactor = np.exp(
            0.5 * ((np.log(2) + 1j * np.pi) * i_bases - loggamma(i_bases + 1))
        ) * np.sqrt(sigma * 2 * np.sqrt(np.pi))
        basis = torch.from_numpy(np.einsum("b, b... -> b...", prefactor, basis_np)).to(
            G.device
        )
        return basis[::2], basis[1::2]  # Split even and odd basis


class Well(Basis):
    """Eigenbasis of infinite well (particle in a box)."""

    def get(self, G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rc = self.r_max  # sharp cutoff (half-width of well)
        k = (
            torch.arange(1, self.n_basis + 1, device=G.device) * np.pi / (2 * rc)
        ).view((-1,) + (1,) * len(G.shape))
        prefactor = -1j * torch.exp(1j * k * rc) * np.sqrt(rc)
        basis = prefactor * torch.special.spherical_bessel_j0(
            (k + G) * rc
        ) + prefactor.conj() * torch.special.spherical_bessel_j0((k - G) * rc)
        return basis[::2], basis[1::2]  # Split even and odd basis


make_basis_map: dict[str, type] = {"hermite": Hermite, "well": Well}


def make_basis(type: str, **kwargs) -> Basis:
    return make_basis_map[type](**kwargs)


def main():
    """Plot basis functions for testing."""
    from ..grid1d import Grid1D, get1D
    import matplotlib.pyplot as plt

    basis = make_basis(type="hermite", n_basis=31, r_max=3.5)
    L = 10.0
    dz = 0.05
    grid1d = Grid1D(L=L, dz=dz)
    z_np = get1D(grid1d.z)
    for label, style, w_tilde in zip(("Even", "Odd"), ("r", "b"), basis(grid1d.grid)):
        w = torch.fft.irfft(w_tilde.detach()).real / dz
        plt.plot(z_np, w.to(qp.rc.cpu).numpy().T, style, label=f"{label} weights")
        overlap = (w @ w.T) * dz
        print(f"\nOverlap matrix for {label} weights:\n{qp.utils.fmt(overlap)}")
    plt.xlim(0, 0.5 * L)
    plt.xlabel("$z$")
    plt.ylabel("$w(z)$")
    plt.show()


if __name__ == "__main__":
    main()

import torch
import qimpy as qp
import numpy as np
from torch.nn.parameter import Parameter
from scipy.special import hermite, loggamma
from functools import cache
from abc import abstractmethod
from typing import Protocol
from .function import Linear


class WeightFunctions(Protocol):
    """Protocol for trainable weight functions."""

    def asdict(self) -> dict:
        """Serialize parameters to dict."""

    def __call__(self, G: torch.Tensor) -> torch.Tensor:
        """Compute weight functions with shape `(n_functions,) + G.shape`.
        Here, `n_functions` is the number of trainable functions,
        which must be one of the keyword inputs to __init__."""


class Gaussian(torch.nn.Module):
    """Gaussian x polynomial weight functions."""

    degree: int  #: degree of polynomial in G^2 multiplying Gaussian
    sigma_max: float  #: maximum value of sigma in Gaussian
    coefficients: torch.Tensor  #: Trainable polynomial coefficients and sigma

    def __init__(self, *, n_functions: int, degree: int, sigma_max: float) -> None:
        super().__init__()
        self.degree = degree
        self.sigma_max = sigma_max
        self.coefficients = Parameter(
            torch.empty((n_functions, degree + 2), device=qp.rc.device)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.coefficients, -1.0, 1.0)

    def asdict(self) -> dict:
        return dict(type="gaussian", degree=self.degree, sigma_max=self.sigma_max)

    def __call__(self, G: torch.Tensor) -> torch.Tensor:
        coefficients, Sigma = self.coefficients.split((self.degree + 1, 1), dim=1)
        bcast_shape = (-1,) + (1,) * len(G.shape)
        sigma = self.sigma_max * torch.special.expit(Sigma).view(bcast_shape)
        sigmaGsq = (sigma * G[None]).square()
        powers = torch.arange(self.degree + 1, device=qp.rc.device).view(bcast_shape)
        polynomial = torch.einsum(
            "fp, fp... -> f...", coefficients, sigmaGsq[:, None] ** powers
        )
        return torch.exp(-0.5 * sigmaGsq) * polynomial


class Basis(torch.nn.Module):
    """Abstract base class for weight functions linearly combining basis sets."""

    n_basis: int  #: number of basis functions of each symmetry (odd / even)
    r_max: float  #: nominal spatial extent of basis (need not be a sharp cutoff)
    coefficients: Linear  #: coefficients linearly combining spline basis

    def __init__(self, *, n_functions: int, n_basis: int, r_max: float) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.r_max = r_max
        self.coefficients = Linear(
            n_basis, n_functions, bias=False, device=qp.rc.device
        )

    def asdict(self) -> dict:
        return dict(
            type=self.__class__.__name__.lower(), n_basis=self.n_basis, r_max=self.r_max
        )

    def __call__(self, G: torch.Tensor) -> torch.Tensor:
        return self.coefficients(self._get_basis(G))

    @abstractmethod
    def _get_basis(self, G: torch.Tensor) -> torch.Tensor:
        """Get basis functions of size (n_basis,) + G.shape."""
        ...


class Hermite(Basis):
    """Eigenbasis of quantum Harmonic oscillator."""

    @property
    def sigma(self) -> float:
        """Gaussian width of ground state."""
        return self.r_max / np.sqrt(4 * self.n_basis)

    @cache
    def _get_basis(self, G: torch.Tensor) -> torch.Tensor:
        # Compute Hermite polynomials (using scipy):
        sigma = self.sigma
        sigmaG = sigma * G.to(qp.rc.cpu).numpy()
        i_bases = np.arange(self.n_basis) * 2  # even functions only
        basis_np = np.array(
            [hermite(i_basis, monic=True)(sigmaG) for i_basis in i_bases]
        )

        # Multiply Gaussian and normalization factors:
        basis_np *= np.exp(-0.5 * np.square(sigmaG))[None]
        prefactor = np.exp(
            0.5 * (np.log(2) * i_bases - loggamma(i_bases + 1))
        ) * np.sqrt(sigma * 2 * np.sqrt(np.pi))
        return torch.from_numpy(np.einsum("b, b... -> b...", prefactor, basis_np)).to(
            G.device
        )


class Well(Basis):
    """Eigenbasis of infinite well (particle in a box)."""

    @cache
    def _get_basis(self, G: torch.Tensor) -> torch.Tensor:
        rc = self.r_max  # sharp cutoff (half-width of well)
        k = (
            torch.arange(1, 2 * self.n_basis, 2, device=G.device) * np.pi / (2 * rc)
        ).view(
            (-1,) + (1,) * len(G.shape)
        )  # corresponding to even functions alone
        prefactor = torch.sin(k * rc) * np.sqrt(rc)
        return prefactor * (
            torch.special.spherical_bessel_j0((k + G) * rc)
            + torch.special.spherical_bessel_j0((k - G) * rc)
        )


class Cspline(Basis):
    """Cubic spline basis (blip functions)."""

    @cache
    def _get_basis(self, G: torch.Tensor) -> torch.Tensor:
        h = self.r_max / (self.n_basis - 1)
        offsets = h * torch.arange(self.n_basis, device=G.device)

        # Construct reference blip basis function at origin:
        n_points = 100  # number of points per spline interval for construction
        dz = h / n_points
        z = dz * torch.arange(0, 2 * n_points + 1, device=G.device)
        t = z / h  # dimensionless 0-to-1 coordinate within each interval
        blip = torch.where(t < 1.0, 1 + 0.75 * t * t * (t - 2), 0.25 * (2 - t) ** 3)
        blip[1:] *= 2.0  # symmetry weight to account for -z
        phase = torch.cos(torch.einsum("z, ... -> z...", z, G))
        blip_tilde = torch.tensordot(blip, phase, dims=1) * dz

        # Move basis functions into place with even symmetry:
        return blip_tilde * torch.cos(torch.einsum("z, ... -> z...", offsets, G))


make_weight_functions_map: dict[str, type] = {
    "gaussian": Gaussian,
    "hermite": Hermite,
    "well": Well,
    "cspline": Cspline,
}


def make_weight_functions(*, type: str, **kwargs) -> Basis:
    return make_weight_functions_map[type](**kwargs)


def main():
    """Plot basis functions for testing."""
    from ..grid1d import Grid1D, get1D
    import matplotlib.pyplot as plt

    weight_functions = make_weight_functions(
        type="cspline", n_functions=1, n_basis=10, r_max=3.5
    )
    L = 10.0
    dz = 0.05
    grid1d = Grid1D(L=L, dz=dz)
    z_np = get1D(grid1d.z)
    w_tilde = weight_functions._get_basis(torch.squeeze(grid1d.Gmag))
    w = torch.fft.irfft(w_tilde.detach()).real / dz
    plt.plot(z_np, w.to(qp.rc.cpu).numpy().T)
    overlap = (w @ w.T) * dz
    print(f"\nOverlap matrix:\n{qp.utils.fmt(overlap)}")
    plt.xlim(0, 0.5 * L)
    plt.xlabel("$z$")
    plt.ylabel("$w(z)$")
    plt.show()


if __name__ == "__main__":
    main()

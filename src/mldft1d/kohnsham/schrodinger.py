import qimpy as qp
import numpy as np
from .. import Grid1D
from ..protocols import get_mu
from typing import Optional
import torch


class Schrodinger:
    """Exact 1D Schrodinger solver"""

    grid1d: Grid1D
    n_bulk: float  #: Bulk number density of the fluid
    mu: float  #: Bulk chemical potential
    T: float  #: Fermi smearing width
    n: qp.grid.FieldR  #: Equilibrium density
    V: qp.grid.FieldR  #: External potential
    energy: qp.Energy  #: Equilibrium energy components

    def __init__(self, grid1d: Grid1D, *, n_bulk: float, T: float) -> None:
        """Initializes to bulk fluid with no external potential."""
        self.grid1d = grid1d
        self.n_bulk = n_bulk
        self.mu = get_mu([ThomasFermi(T)], n_bulk)
        self.T = T
        self.n = qp.grid.FieldR(grid1d.grid, data=torch.zeros_like(grid1d.z))
        self.V = self.n.zeros_like()
        self.energy = qp.Energy()

    def known_V(self) -> Optional[qp.grid.FieldR]:
        """No known potential: learn the entire part in ML."""
        return None

    def minimize(self) -> qp.Energy:
        Nk = 2 * np.ceil(2 * np.pi / (self.grid1d.L * self.T))  # assuming vF ~ 1
        k = np.arange(Nk // 2 + 1) * (1.0 / Nk)  # in fractional coords, symm reduced
        wk = np.where(k, 2, 1) * (2.0 / Nk)  # weight of each k-point

        # Accumulate contributions by k:
        energy = self.energy
        energy.update(KE=0.0, Ext=0.0)
        self.n.data.zero_()
        for ki, wki in zip(k, wk):
            self.accumulate_k(ki, wki)

        # Move V.n energy from KE to Ext component:
        V_dot_n = self.V ^ self.n
        energy["KE"] -= V_dot_n
        energy["Ext"] += V_dot_n
        return energy

    def accumulate_k(self, k: float, wk: float) -> None:
        """Solve for one k-point, accumulating energy and density contributions."""
        L = self.grid1d.L
        Nz = self.grid1d.z.shape[2]

        # Full KE operator:
        iG = self.grid1d.grid.get_mesh("G")[..., 2].flatten()
        k_plus_G = (iG + k) * (2 * np.pi / L)
        KE_diag = 0.5 * (k_plus_G**2)

        # Reduced KE for plane-wave basis:
        Gmax_rho = (np.pi / L) * Nz  # Nyquist frequency for charge density grid
        Gmax_wfn = 0.5 * Gmax_rho
        KEcut_wfn = 0.5 * (Gmax_wfn**2)
        sel = torch.where(KE_diag <= KEcut_wfn)[0]
        iG = iG[sel]
        KE = torch.diag(KE_diag[sel])
        n_bands = len(sel)

        # Potential operator in PW basis
        V = self.V.data.flatten()
        Vtilde = torch.fft.ifft(V)
        ij_grid = (iG[:, None] - iG[None, :]) % Nz
        Vop = Vtilde[ij_grid]
        H = KE + Vop

        # Diagonalize the Hamiltonian to find the eigenvalues and eigenvectors
        eig, psi_reduced = torch.linalg.eigh(H)
        f = torch.special.expit((self.mu - eig) / self.T)  # Fermi-Dirac occupations
        fbar = 1.0 - f
        S = -torch.special.xlogy(f, f) - torch.special.xlogy(fbar, fbar)  # Entropy

        # Compute energy components:
        energy = self.energy
        energy["KE"] += wk * (eig @ f - self.T * S.sum())  # Includes V.n for now,
        energy["Ext"] -= wk * (self.mu * f.sum())  # ... which is moved here later

        # Compute density contributions:
        psi_tilde = torch.zeros(
            (n_bands, Nz), dtype=psi_reduced.dtype, device=qp.rc.device
        )
        psi_tilde[:, iG] = psi_reduced.T
        psi_sqr = qp.utils.abs_squared(torch.fft.fft(psi_tilde))
        self.n.data += (wk / L) * (f @ psi_sqr)[None, None, :]


class ThomasFermi:
    """Bulk kinetic energy density functional, with optional temperature correction."""

    prefactor: float
    T: float  #: Temperature (Fermi smearing)

    def __init__(self, T: float = 0.0) -> None:
        self.prefactor = (np.pi**2) / 24
        self.T = T

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        e = self.prefactor * (n**3)  # Thomas-Fermi (T=0) part in 1D
        if self.T:
            e += n * self.energy_correction_per_particle(n)
        return e

    def energy_correction_per_particle(self, n: torch.Tensor) -> torch.Tensor:
        x = n.square() / (2 * self.T)
        x_sq = x.square()
        low_density = 2.0 - (np.pi * x).log() - 1.311 * x.sqrt()
        numerator = low_density + x * (1.600 + x_sq * (0.697 + x_sq * 0.279))
        denominator = 1.0 + x_sq * (0.596 + x_sq * (1.018 + x_sq * (0.279 * 1.5)))
        return (-0.5 * self.T) * numerator / denominator

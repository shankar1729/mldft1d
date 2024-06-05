from __future__ import annotations

import numpy as np
import torch

from qimpy import rc, log, Energy
from qimpy.math import abs_squared
from qimpy.grid import FieldR
from .. import Grid1D
from ..protocols import get_mu
from ..hf import SolveNumberConstraint


class Schrodinger:
    """Exact 1D Schrodinger solver"""

    grid1d: Grid1D
    n_bulk: torch.Tensor  #: Bulk number density of the fluid
    T: float  #: Fermi smearing width
    mu: torch.Tensor  #: Bulk chemical potential
    fixed_number: bool  #: If yes, switch to canonical mode (else grand-canonical)
    n: FieldR  #: Equilibrium density
    V: FieldR  #: External potential
    energy: Energy  #: Equilibrium energy components

    k: torch.Tensor  #: reduced k-points
    wk: float  #: k-point weights
    iG: torch.Tensor  #: basis coefficients (common for all k)
    ke: torch.Tensor  #: kinetic energy of each k-point and G-vector

    def __init__(
        self,
        grid1d: Grid1D,
        *,
        n_bulk: torch.Tensor,
        T: float,
        fixed_number: bool = False,
        Lsup: float = 200.0,
    ) -> None:
        """Initializes to bulk fluid with no external potential."""
        self.grid1d = grid1d
        self.n_bulk = n_bulk
        self.T = T
        self.mu = (
            torch.zeros_like(n_bulk)
            if fixed_number
            else get_mu([ThomasFermi(T)], n_bulk)
        )
        self.fixed_number = fixed_number
        self.n = FieldR(
            grid1d.grid, data=torch.zeros((1,) + grid1d.z.shape, device=rc.device)
        )
        self.V = self.n.zeros_like()
        self.energy = Energy()

        # Setup k-points:
        Nk = 2 * int(np.ceil(0.5 * Lsup / self.grid1d.L))
        dk = 1.0 / Nk
        self.k = torch.arange(0.5 * dk, 0.5, dk, device=rc.device)  # off-Gamma
        self.wk = 4 * dk  # weight of each k-point (2 for symm, 2 for spin)
        log.info(f"Reduced {Nk} k-points to {Nk // 2} using inversion symmetry")

        # Setup basis and kinetic energy operator:
        iG_full = grid1d.grid.get_mesh("G")[..., 2].flatten()
        Nz = len(iG_full)
        self.iG = iG_full[abs(iG_full) <= Nz // 4]  # wavefunction iG
        log.info(f"Initialized common basis with {len(self.iG)} plane waves/k-point")
        self.ke = (
            0.5
            * ((self.k[:, None] + self.iG[None, :]) * (2 * np.pi / grid1d.L)).square()
        )

    def training_targets(self) -> tuple[float, FieldR]:
        KE = float(self.energy["KE"])
        V_kinetic = FieldR(self.V.grid, data=(self.mu.view(-1, 1, 1, 1) - self.V.data))
        return KE, V_kinetic

    def minimize(self) -> Energy:
        # Construct kinetic Hamiltonian for all k:
        H = torch.diag_embed(self.ke).to(torch.complex128)

        # Add potential operator in reciprocal space:
        Nz = self.grid1d.grid.shape[2]
        Vtilde = torch.fft.fft(self.V.data.flatten(), norm="forward")
        H += Vtilde[(self.iG[:, None] - self.iG[None, :]) % Nz]

        # Diagonalize:
        eig, evecs = torch.linalg.eigh(H)
        evecs = evecs.swapaxes(-2, -1)  # band dimensions first, then basis

        # Update fillings:
        if self.fixed_number:
            f_sum = self.n_bulk * self.grid1d.L / self.wk
            mu = SolveNumberConstraint.apply(eig, self.T, f_sum)
        else:
            mu = self.mu
        f = torch.special.expit((mu - eig) / self.T)  # Fermi-Dirac occupations

        # Truncate unoccupied bands:
        f_cut = 1e-15
        Nbands = torch.where(f.max(dim=0).values > f_cut)[0][-1] + 1
        f = f[:, :Nbands]
        eig = eig[:, :Nbands]
        evecs = evecs[:, :Nbands]

        # Update density:
        Nk, Nbands, NG = evecs.shape
        psi = torch.zeros((Nk, Nbands, Nz), dtype=torch.complex128, device=rc.device)
        psi[..., self.iG] = evecs  # G-sphere to full reciprocal grid
        psi = torch.fft.ifft(psi, norm="forward")  # real-space wavefunctions
        self.n.data = (self.wk / self.grid1d.L) * torch.einsum(
            "kb, kbz -> z", f, abs_squared(psi)
        ).view(1, 1, 1, -1)

        # Set energy components:
        energy = self.energy
        V_dot_n = (self.V ^ self.n).sum(dim=-1)
        energy["Ext"] = V_dot_n - self.wk * self.mu * f.sum()
        energy["KE"] = self.wk * (eig * f).sum() - V_dot_n
        fbar = 1.0 - f
        S = -torch.special.xlogy(f, f) - torch.special.xlogy(fbar, fbar)  # Entropy
        energy["-TS"] = -self.T * S.sum() * self.wk
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
            (n_bands, Nz), dtype=psi_reduced.dtype, device=rc.device
        )
        psi_tilde[:, iG] = psi_reduced.T
        psi_sqr = abs_squared(torch.fft.fft(psi_tilde))
        self.n.data += (wk / L) * (f @ psi_sqr)[None, None, :]


class ThomasFermi:
    """Bulk kinetic energy density functional, with optional temperature correction."""

    prefactor: float
    T: float  #: Temperature (Fermi smearing)

    def __init__(self, T: float = 0.0) -> None:
        self.prefactor = (np.pi**2) / 24
        self.T = T

    def __call__(self, n: torch.Tensor) -> torch.Tensor:
        """Underlying elementwise calculator prior to any dimension contractions."""
        e = self.prefactor * (n**3)  # Thomas-Fermi (T=0) part in 1D
        if self.T:
            e += n * self.energy_correction_per_particle(n)
        return e

    def get_energy(self, n: FieldR) -> torch.Tensor:
        return FieldR(n.grid, data=self(n.data)).integral().sum(dim=0)

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        return self(n).sum(dim=-1)

    def energy_correction_per_particle(self, n: torch.Tensor) -> torch.Tensor:
        x = n.square() / (2 * self.T)
        x_sq = x.square()
        low_density = 2.0 - (np.pi * x).log() - 1.311 * x.sqrt()
        numerator = low_density + x * (1.600 + x_sq * (0.697 + x_sq * 0.279))
        denominator = 1.0 + x_sq * (0.596 + x_sq * (1.018 + x_sq * (0.279 * 1.5)))
        return (-0.5 * self.T) * numerator / denominator

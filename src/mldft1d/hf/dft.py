from __future__ import annotations
from typing import Optional, Sequence, Union

import torch
import numpy as np

from qimpy import log, rc, MPI, Energy
from qimpy.io import CheckpointPath
from qimpy.grid import FieldR, FieldH
from qimpy.algorithms import Pulay
from qimpy.math import abs_squared

from .. import Grid1D
from ..protocols import get_mu, Functional
from ..kohnsham import ThomasFermi
from . import SoftCoulomb, BulkExchange


class DFT:
    """Density-functional / Hartree-Fock solver"""

    grid1d: Grid1D
    exchange_functional: Optional[Functional]  #: Ex[n], and Exx[psi] if None
    n_bulk: float  #: Bulk number density of the fluid
    n_electrons: float  #: Electron number/cell: overrides n_bulk if non-zero
    bulk_exchange: BulkExchange  #: Bulk exchange energy density
    soft_coulomb: SoftCoulomb  #: 1D Soft-Coulomb interaction kernel
    mu: float  #: Bulk chemical potential (fixed when n_electrons is zero)
    T: float  #: Fermi smearing width
    n: FieldR  #: Equilibrium density
    V: FieldR  #: External potential
    scf: SCF  #: Self-consistent field algorithm (for LDA/ML cases only)

    energy: Energy  #: Equilibrium energy components
    eig: torch.Tensor  #: Kohn-Sham eigenvalues
    f: torch.Tensor  #: Fermi occupations
    C: torch.Tensor  #: Wavefunctions
    k: torch.Tensor  #: reduced k-points
    wk: float  #: k-point weights
    iG: torch.Tensor  #: basis coefficients (common for all k)
    coulomb_tilde: torch.Tensor  #: Coulomb kernel in reciprocal space
    ke: torch.Tensor  #: kinetic energy of each k-point and G-vector

    def __init__(
        self,
        grid1d: Grid1D,
        *,
        exchange_functional: Union[str, Functional],
        n_bulk: float,
        T: float,
        n_electrons: float = 0.0,
    ) -> None:
        comm = grid1d.grid.comm
        assert (comm is None) or (comm.size == 1)  # No MPI support here for simplicity
        self.grid1d = grid1d
        self.n_bulk = n_bulk
        self.bulk_exchange = BulkExchange()
        self.soft_coulomb = SoftCoulomb()
        self.mu = get_mu([ThomasFermi(T), self.bulk_exchange], n_bulk)
        log.info(f"Set electron chemical potential: {self.mu:.5f}")
        self.T = T
        self.n = FieldR(grid1d.grid, data=torch.zeros_like(grid1d.z))
        self.V = self.n.zeros_like()
        self.energy = Energy()
        if isinstance(exchange_functional, str):
            assert exchange_functional in {"lda", "exact"}
            if exchange_functional == "lda":
                self.exchange_functional = self.bulk_exchange  # LDA
            else:
                self.exchange_functional = None  # Evaluate EXX explicitly
        else:  # Externally specified functional, eg. for ML approximation
            self.exchange_functional = exchange_functional

        # Separate initialization for EXX and density functionals:
        if self.exchange_functional is None:
            raise NotImplementedError  # TODO: Initialize EXX-OEP
        else:
            self.scf = SCF(self)

        # Setup k-points:
        Nk = 2 * int(np.ceil(2 * np.pi / (self.grid1d.L * self.T)))  # assuming vF ~ 1
        dk = 1.0 / Nk
        self.k = torch.arange(0.5 * dk, 0.5, dk, device=rc.device)  # Monkhorst-Pack
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

        # Initialize coulomb kernel for Hartree term:
        self.coulomb_tilde = self.soft_coulomb.tilde(grid1d.Gmag.flatten())
        self.coulomb_tilde[0] = 0.0

        # Staring point analgous to LCAO:
        self.n.data[:] = n_bulk
        self.update_potential()
        self.diagonalize()

        mu_tmp = self.eig[:, 2].max()
        self.mu, mu_tmp = mu_tmp, self.mu  # HACK: make sure some electrons present
        self.update()
        self.mu, mu_tmp = mu_tmp, self.mu

    def update_density(self) -> None:
        """Update electron density from wavefunctions and fillings."""
        C = self.C
        Nk, Nbands, NG = C.shape
        Nz = self.grid1d.grid.shape[2]
        Cfull = torch.zeros((Nk, Nbands, Nz), dtype=torch.complex128, device=rc.device)
        Cfull[..., self.iG] = C  # G-sphere to full reciprocal grid
        self.n.data = (self.wk / self.grid1d.L) * torch.einsum(
            "kb, kbz -> z", self.f, abs_squared(torch.fft.fft(Cfull))
        )[None, None]

    def update_potential(self, requires_grad: bool = True) -> None:
        """Update density-dependent energy terms and electron potential.
        If `requires_grad` is False, only compute the energy (skip the potentials)."""
        n = self.n
        if requires_grad:
            n.data.requires_grad = True
            n.data.grad = None
        energy = self.energy
        energy["EH"] = 0.5 * (n ^ n.convolve(self.coulomb_tilde))
        energy["Eext"] = n ^ self.V
        if self.exchange_functional is not None:
            energy["Ex"] = self.exchange_functional.get_energy(n)
        if requires_grad:
            E = energy.sum_tensor()
            assert E is not None
            E.backward()  # partial derivative dE/dn -> self.n.data.grad
            n.data.requires_grad = False

    def update_fillings(self):
        """Update fillings and drop unoccupied bands in f and C."""
        # TODO: support fixed N mode (grand-canonical only so far)
        f = torch.special.expit((self.mu - self.eig) / self.T)  # Fermi-Dirac fillings
        fbar = 1.0 - f
        S = -torch.special.xlogy(f, f) - torch.special.xlogy(fbar, fbar)  # Entropy
        self.energy["-TS"] = -self.T * S.sum() * self.wk
        self.energy["-muN"] = -self.mu * f.sum() * self.wk

        n_electrons = self.wk * f.sum().item()
        log.info(
            f"  FillingsUpdate:  mu: {self.mu:.9f}  n_electrons: {n_electrons:.6f}"
        )

        # Drop unoccupied bands
        f_cut = 1e-15
        Nbands = torch.where(f.max(dim=0).values > f_cut)[0][-1] + 1
        self.f = f[:, :Nbands]
        self.C = self.C[:, :Nbands]

    def update(self, requires_grad: bool = True) -> None:
        """Update electronic system to current wavefunctions and eigenvalues.
        This updates occupations, density, potential and electronic energy.
        If `requires_grad` is False, only compute the energy (skip the potentials)."""
        self.update_fillings()
        self.update_density()
        self.update_potential(requires_grad)
        self.energy["KE"] = self.wk * torch.einsum(
            "kb, kG, kbG ->", self.f, self.ke, abs_squared(self.C)
        )

    def diagonalize(self) -> None:
        """Compute eigenvectors C and eigenvalues eig in current potential `n.grad`"""
        H = torch.diag_embed(self.ke).to(torch.complex128)
        # Add potential operator in reciprocal space
        assert self.n.data.grad is not None
        Vks = self.n.data.grad.flatten() / self.grid1d.grid.dV  # convert from dE/dn
        Vks_tilde = torch.fft.ifft(Vks)
        Nz = len(Vks)
        ij_grid = (self.iG[:, None] - self.iG[None, :]) % Nz
        H += Vks_tilde[ij_grid]
        # Diagonalize
        self.eig, self.C = torch.linalg.eigh(H)

    def minimize(self) -> Energy:
        if self.exchange_functional is None:
            raise NotImplementedError  # TODO: Implement EXX-OEP
        else:
            self.scf.optimize()
        log.info(f"\nEnergy components:\n{self.energy}\n")
        return self.energy

    def known_part(self) -> Optional[tuple[float, FieldR]]:
        """Known part of equilibrium energy and potential eg. ideal gas, if any."""
        return NotImplemented


class SCF(Pulay[FieldH]):
    dft: DFT
    K_kerker: torch.Tensor  #: Kerker mixing kernel
    K_metric: torch.Tensor  #: Pulay metric kernel

    def __init__(
        self,
        dft: DFT,
        *,
        n_iterations: int = 50,
        energy_threshold: float = 1e-8,
        residual_threshold: float = 1e-7,
        n_consecutive: int = 2,
        n_history: int = 10,
        mix_fraction: float = 0.5,
        q_kerker: float = 0.8,
        q_metric: float = 0.8,
        eig_threshold: float = 1e-8,
    ):
        self.dft = dft
        super().__init__(
            comm=MPI.COMM_SELF,
            name="SCF",
            checkpoint_in=CheckpointPath(),
            n_iterations=n_iterations,
            energy_threshold=float(energy_threshold),
            residual_threshold=float(residual_threshold),
            n_consecutive=n_consecutive,
            extra_thresholds={"|deig|": eig_threshold},
            n_history=n_history,
            mix_fraction=mix_fraction,
        )
        self.initialize_kernels(q_kerker, q_metric)

    def cycle(self, dEprev: float) -> Sequence[float]:
        dft = self.dft
        Nbands = dft.f.shape[1]
        eig_prev = dft.eig[:, :Nbands]
        dft.diagonalize()
        dft.update()  # update total energy
        # Compute eigenvalue difference for extra convergence threshold:
        eig_cur = dft.eig[..., :Nbands]
        deig_max = (eig_cur - eig_prev).abs().max().item()
        return [deig_max]

    def initialize_kernels(self, q_kerker: float, q_metric: float) -> None:
        """Initialize preconditioner and metric."""
        grid = self.dft.grid1d.grid
        iG = grid.get_mesh("H").to(torch.double)  # half-space
        wG = grid.lattice.volume * grid.weight2H  # integration weight
        Gsq = ((iG @ grid.lattice.Gbasis.T) ** 2).sum(dim=-1)
        Gsq_reg = torch.clamp(Gsq, min=Gsq[Gsq > 0.0].min())  # regularized
        self.K_kerker = (Gsq_reg + 0.01) / (Gsq_reg + q_kerker**2)
        self.K_metric = (1.0 + q_metric**2 / Gsq_reg) * wG

    @property
    def energy(self) -> Energy:
        return self.dft.energy

    @property
    def variable(self) -> FieldH:
        return ~self.dft.n

    @variable.setter
    def variable(self, n_tilde: FieldH) -> None:
        self.dft.n = ~n_tilde
        self.dft.update_potential()

    def precondition(self, v: FieldH) -> FieldH:
        return v.convolve(self.K_kerker)

    def metric(self, v: FieldH) -> FieldH:
        return v.convolve(self.K_metric)

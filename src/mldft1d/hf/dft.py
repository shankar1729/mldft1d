from __future__ import annotations
from typing import Optional, Sequence, Union
from itertools import chain

import torch
import numpy as np
from scipy.optimize import brentq

from qimpy import log, rc, MPI, Energy
from qimpy.io import CheckpointPath
from qimpy.grid import FieldR, FieldH
from qimpy.algorithms import Pulay
from qimpy.math import abs_squared

from .. import Grid1D
from ..protocols import Functional
from . import SoftCoulomb, BulkExchange
from .oep import OEP


class DFT:
    """Density-functional / Hartree-Fock solver"""

    grid1d: Grid1D
    exchange_functional: Optional[Functional]  #: Ex[n], and Exx[psi] if None
    n_bulk: float  #: Bulk number density of the fluid
    n_electrons: float  #: Electron number/cell
    bulk_exchange: BulkExchange  #: Bulk exchange energy density
    soft_coulomb: SoftCoulomb  #: 1D Soft-Coulomb interaction kernel
    mu: float  #: Bulk chemical potential (fixed when n_electrons is zero)
    T: float  #: Fermi smearing width
    n: FieldR  #: Equilibrium density
    V: FieldR  #: External potential
    scf: SCF  #: Self-consistent field algorithm (for LDA/ML cases only)
    oep: OEP

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
        a: float = 1.0,
    ) -> None:
        comm = grid1d.grid.comm
        assert (comm is None) or (comm.size == 1)  # No MPI support here for simplicity
        self.grid1d = grid1d
        self.n_bulk = n_bulk
        self.bulk_exchange = BulkExchange(a)
        self.soft_coulomb = SoftCoulomb(a)
        self.n_electrons = n_bulk * grid1d.L
        self.T = T
        self.n = FieldR(grid1d.grid, data=torch.ones_like(grid1d.z) * n_bulk)
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
            self.oep = OEP(self)
        else:
            self.scf = SCF(self)

        # Print params to output
        log.info(f"L: {self.grid1d.L}")
        log.info(f"dz: {self.grid1d.dz}")
        log.info(f"T: {self.T}")
        log.info(f"n_bulk: {self.n_bulk}")
        log.info(f"a: {a}")

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

        # Initialize truncated kernel for exact exchange:
        Nz_sup = Nz * Nk
        scale = np.ceil(3 * grid1d.dz / a)
        dz_fine = grid1d.dz / scale
        Nz_sup_fine = Nz_sup * scale
        z_sup_fine = dz_fine * torch.cat(
            (
                torch.arange(Nz_sup_fine // 2, device=rc.device),
                torch.arange(Nz_sup_fine // 2 - Nz_sup_fine, 0, device=rc.device),
            )
        )
        Ksup_fine = self.soft_coulomb(z_sup_fine)
        Ksup_fine_tilde = torch.fft.fft(Ksup_fine).real * dz_fine
        Ksup_tilde = torch.cat(
            (Ksup_fine_tilde[: Nz_sup // 2], Ksup_fine_tilde[-Nz_sup // 2 :])
        )
        # --- split supercell kernel by integer q offsets
        self.Kx_tilde = torch.zeros((2 * Nk - 1, Nz))
        self.Kx_tilde[0] = Ksup_tilde[::Nk]
        for iq in range(1, Nk):
            Kx_cur = Ksup_tilde[iq::Nk]
            self.Kx_tilde[iq] = Kx_cur
            self.Kx_tilde[iq - Nk] = torch.roll(Kx_cur, 1)

    def to_real_space(self, C: torch.Tensor) -> torch.Tensor:
        Nk, Nbands, NG = C.shape
        Nz = self.grid1d.grid.shape[2]
        Cfull = torch.zeros((Nk, Nbands, Nz), dtype=torch.complex128, device=rc.device)
        Cfull[..., self.iG] = C  # G-sphere to full reciprocal grid
        return torch.fft.ifft(Cfull, norm="forward")

    def update_density(self) -> None:
        """Update electron density from wavefunctions and fillings."""
        self.n.data = (self.wk / self.grid1d.L) * torch.einsum(
            "kb, kbz -> z", self.f, abs_squared(self.to_real_space(self.C))
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
        mu = SolveNumberConstraint.apply(self.eig, self.T, self.n_electrons / self.wk)
        FTOL = 1e-16
        f = torch.special.expit((mu - self.eig) / self.T)
        fbar = 1.0 - f
        S = -torch.special.xlogy(f, f.clamp(min=FTOL)) - torch.special.xlogy(
            fbar, fbar.clamp(min=FTOL)
        )  # Entropy
        self.energy["-TS"] = -self.T * S.sum() * self.wk
        self.mu = mu.item()
        log.info(
            f"  FillingsUpdate:  mu: {self.mu:.9f}  n_electrons: {self.n_electrons:.6f}"
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
        if self.exchange_functional is None:
            self.energy["Ex"] = self.compute_exx(self.C, self.f)

    def diagonalize(self, Vks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvectors C and eigenvalues eig in current potential `n.grad`"""
        H = torch.diag_embed(self.ke).to(torch.complex128)
        # Add potential operator in reciprocal space
        Vks_tilde = torch.fft.fft(Vks, norm="forward")
        Nz = len(Vks)
        ij_grid = (self.iG[:, None] - self.iG[None, :]) % Nz
        H += Vks_tilde[ij_grid]
        # Diagonalize
        eig, evecs = torch.linalg.eigh(H)
        return evecs.swapaxes(-2, -1), eig

    def initialize_state(self) -> None:
        # Staring point analgous to LCAO:
        self.update_potential()
        Vks = self.n.data.grad.flatten() / self.grid1d.grid.dV  # convert from dE/dn
        self.C, self.eig = self.diagonalize(Vks)
        self.mu = self.eig.min().item()  # initial value before update_fillings
        self.update_fillings()

    def compute_exx(self, C: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        k = self.k
        dk = (k[1] - k[0]).item()
        IC = self.to_real_space(C)
        Exx = torch.zeros(tuple(), device=rc.device)
        for k1, IC1, f1 in zip(chain(k, -k), chain(IC, IC.conj()), chain(f, f)):
            # for k2, IC2, f2 in zip(k, IC, f):
            iq = torch.round((k - k1.item()) / dk).to(int)
            K = self.Kx_tilde[iq]
            n12 = torch.einsum("ax, kbx -> kabx", IC1.conj(), IC)
            n12tilde = torch.fft.fft(n12, norm="forward")
            Exx += torch.einsum("kabG, kG, a, kb ->", abs_squared(n12tilde), K, f1, f)
        wk_unreduced = dk  # without spin and symmetry weights
        return (-0.5 * wk_unreduced * self.wk / self.grid1d.L) * Exx

    def minimize(self) -> Energy:
        if not hasattr(self, "C"):
            log.info("\n(Initializing state)\n")
            self.initialize_state()
        if self.exchange_functional is None:
            log.info("\nBeginning effective-potential optimizer\n")
            self.oep.optimize()
        else:
            log.info("\nBeginning self-consistent field\n")
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
        n_iterations: int = 500,
        energy_threshold: float = 1e-8,
        residual_threshold: float = 1e-7,
        n_consecutive: int = 2,
        n_history: int = 15,
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
        Vks = dft.n.data.grad.flatten() / dft.grid1d.grid.dV  # convert from dE/dn
        dft.C, dft.eig = dft.diagonalize(Vks)
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
        Gmag = ((iG @ grid.lattice.Gbasis.T) ** 2).sum(dim=-1).sqrt()
        Gmag_reg = torch.clamp(Gmag, min=Gmag[Gmag > 0.0].min())  # regularized
        invCoul = Gmag_reg**2  # self.dft.soft_coulomb.tilde(Gmag_reg) / (4 * np.pi)#
        self.K_kerker = invCoul / (invCoul + q_kerker**2)
        self.K_metric = (1.0 + q_metric**2 / invCoul) * wG

    @property
    def energy(self) -> Energy:
        return self.dft.energy

    @property
    def variable(self) -> FieldH:
        return ~self.dft.n

    @variable.setter
    def variable(self, n_tilde: FieldH) -> None:
        self.dft.n = ~n_tilde
        self.dft.n.data.clip_(min=1.0e-15)
        self.dft.update_potential()

    def precondition(self, v: FieldH) -> FieldH:
        return v.convolve(self.K_kerker)

    def metric(self, v: FieldH) -> FieldH:
        return v.convolve(self.K_metric)


class SolveNumberConstraint(torch.autograd.Function):
    """..."""

    @staticmethod
    def forward(ctx, eig: torch.Tensor, T: float, fsum0: float) -> torch.Tensor:  # type: ignore
        def n_electron_error(mu: float) -> float:
            f = torch.special.expit((mu - eig) / T)
            return f.sum().item() - fsum0

        def expand_range(sign: int) -> float:
            mu_limit = eig.mean().item()
            dmu = sign * T
            while sign * n_electron_error(mu_limit) <= 0.0:
                mu_limit += dmu
                dmu *= 2.0
            return mu_limit

        mu_min = expand_range(-1)
        mu_max = expand_range(+1)
        mu = brentq(n_electron_error, mu_min, mu_max)
        ctx.eig = eig
        ctx.T = T
        ctx.mu = mu
        return torch.tensor(mu, device=rc.device)

    @staticmethod
    def backward(ctx, grad_mu: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # type: ignore
        f = torch.special.expit((ctx.mu - ctx.eig) / ctx.T)
        f_prime = f * (f - 1.0) / ctx.T
        mu_prime = f_prime / f_prime.sum()
        return grad_mu * mu_prime, None, None

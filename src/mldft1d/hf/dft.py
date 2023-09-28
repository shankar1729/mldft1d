from __future__ import annotations
from typing import Optional, Sequence, Union

import torch
import numpy as np

from qimpy import log, rc, MPI, Energy
from qimpy.io import CheckpointPath
from qimpy.grid import FieldR, FieldH
from qimpy.algorithms import Pulay

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
    energy: Energy  #: Equilibrium energy components
    eig: torch.Tensor  #: Kohn-Sham eigenvalues
    f: torch.Tensor  #: Fermi occupations
    scf: SCF  #: Self-consistent field algorithm (for LDA/ML cases only)

    k: torch.Tensor  #: reduced k-points
    wk: float  #: k-point weights
    iG: torch.Tensor  #: basis coefficients (common for all k)

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

        # Setup basis:
        iG_full = grid1d.grid.get_mesh("G")[..., 2].flatten()
        Nz = len(iG_full)
        self.iG = iG_full[abs(iG_full) <= Nz // 4]  # wavefunction iG
        log.info(f"Initialized common basis with {len(self.iG)} plane waves/k-point")

    def minimize(self) -> Energy:
        if self.exchange_functional is None:
            raise NotImplementedError  # TODO: Implement EXX-OEP
        else:
            self.scf.optimize()
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
        """
        electrons = self.system.electrons
        eig_prev = electrons.eig[..., : electrons.fillings.n_bands]
        eig_threshold_inner = min(1e-6, 0.1 * abs(dEprev))
        electrons.diagonalize(
            n_iterations=self.n_eig_steps, eig_threshold=eig_threshold_inner
        )
        electrons.update(self.system)  # update total energy
        # Compute eigenvalue difference for extra convergence threshold:
        eig_cur = electrons.eig[..., : electrons.fillings.n_bands]
        deig = (eig_cur - eig_prev).abs()
        deig_max = globalreduce.max(deig, electrons.comm)
        return [deig_max]
        """
        return NotImplemented

    def initialize_kernels(self, q_kerker: float, q_metric: float) -> None:
        """Initialize preconditioner and metric."""
        grid = self.dft.grid1d.grid
        iG = grid.get_mesh("H").to(torch.double)  # half-space
        wG = grid.lattice.volume * grid.weight2H  # integration weight
        Gsq = ((iG @ grid.lattice.Gbasis.T) ** 2).sum(dim=-1)
        Gsq_reg = torch.clamp(Gsq, min=Gsq[Gsq > 0.0].min())  # regularized
        self.K_kerker = Gsq_reg / (Gsq_reg + q_kerker**2)
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

    def precondition(self, v: FieldH) -> FieldH:
        return v.convolve(self.K_kerker)

    def metric(self, v: FieldH) -> FieldH:
        return v.convolve(self.K_metric)

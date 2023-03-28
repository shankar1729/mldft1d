import qimpy as qp
import numpy as np
from mldft1d import Grid1D
import torch


class Schrodinger:
    """Exact 1D Schrodinger solver"""

    grid1d: Grid1D
    T: float  #: Fermi smearing width
    n_bulk: float  #: Bulk number density of the fluid
    mu: float  #: Bulk chemical potential
    e_bulk: float  #: Bulk energy density
    logn: qp.grid.FieldR  #: State of the classical DFT (effectively local mu)
    V: qp.grid.FieldR  #: External potential

    def __init__(self, grid1d: Grid1D, *, n_bulk: float, T: float) -> None:
        """Initializes to bulk fluid with no external potential."""
        self.grid1d = grid1d
        self.T = T
        self.set_n_bulk(n_bulk)
        self.n = qp.grid.FieldR(grid1d.grid, data=torch.zeros_like(grid1d.z))
        self.V = self.n.zeros_like()

    def set_n_bulk(self, n_bulk: float) -> None:
        """Update the chemical potential to target given bulk density."""
        self.n_bulk = n_bulk
        self.e_bulk = (np.pi**2) * (n_bulk**3) / 24
        self.mu = (np.pi * n_bulk) ** 2 / 8

    def minimize(self):
        Nk = 2 * np.ceil(2 * np.pi / (self.grid1d.L * self.T))  # assuming vF ~ 1
        k = np.arange(Nk // 2 + 1) * (1.0 / Nk)  # in fractional coords, symm reduced
        wk = np.where(k, 2, 1) * (2.0 / Nk)  # weight of each k-point
        E = 0.0
        self.n.data.zero_()
        for ki, wki in zip(k, wk):
            Ek, rho_k = self.solve_k(ki)
            E += wki * Ek
            self.n.data += wki * rho_k[None, None, :]
        return E

    def solve_k(self, k) -> tuple[float, np.ndarray]:
        """Solve for one k-point, returning energy and density contributions."""
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
        G = eig @ f - self.T * S.sum() - self.mu * f.sum()  # Grand free energy

        # Expand wavefunctions for density compute:
        psi_tilde = torch.zeros(
            (n_bands, Nz), dtype=psi_reduced.dtype, device=qp.rc.device
        )
        psi_tilde[:, iG] = psi_reduced.T

        psi_sqr = qp.utils.abs_squared(torch.fft.fft(psi_tilde))
        rho = (f @ psi_sqr) / L
        return G, rho

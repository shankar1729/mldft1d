import qimpy as qp
import numpy as np
import hardrods1d as hr
import torch


class Schrodinger:
    """Exact 1D Schrodinger solver"""

    grid1d: hr.Grid1D
    T: float  #: Fermi smearing width
    n_bulk: float  #: Bulk number density of the fluid
    mu: float  #: Bulk chemical potential
    e_bulk: float  #: Bulk energy density
    logn: qp.grid.FieldR  #: State of the classical DFT (effectively local mu)
    V: qp.grid.FieldR  #: External potential

    def __init__(self, grid1d: hr.Grid1D, *, n_bulk: float, T: float) -> None:
        """Initializes to bulk fluid with no external potential."""
        self.grid1d = grid1d
        self.T = T
        self.set_n_bulk(n_bulk)
        self.n = qp.grid.FieldR(grid1d.grid, data=torch.zeros_like(grid1d.z))
        self.V = self.n.zeros_like()

    def set_n_bulk(self, n_bulk: float) -> None:
        """Update the chemical potential to target given bulk density."""
        self.n_bulk = n_bulk
        # Bulk free energy density, omega(n) = nT(log(n) - log(1-2Rn)) - mu n
        # Derivative domega/dn = T [log(n) - log(1-2Rn) + 1 + 2Rn/(1-2Rn)] - mu = 0
        self.mu = (np.pi * n_bulk) ** 2 / 8

    def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
        self.logn += step_size * direction

    def finite_difference_test(*args, **kwargs):
        ...

    def minimize(self):
        N_k_points = 100
        k_points = np.arange(N_k_points) * (1.0 / N_k_points)  # in fractional coords
        wk = 2.0 / N_k_points  # weight of each k-point (including spin degeneracy)
        E = 0.0
        self.n.data.zero_()
        for i, k in enumerate(k_points):
            Ek, rho_k = self.solve_k(k)
            E += wk * Ek
            self.n.data += wk * rho_k[None, None, :]
        return E

    def solve_k(self, k) -> tuple[float, np.ndarray]:
        """Solve for one k-point, returning energy and density contributions."""
        L = self.grid1d.L
        dz = self.grid1d.dz
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
        E = eig @ f  # Total energy

        # Expand wavefunctions for density compute:
        psi_tilde = torch.zeros(
            (n_bands, Nz), dtype=psi_reduced.dtype, device=qp.rc.device
        )
        psi_tilde[:, iG] = psi_reduced.T

        psi_sqr = qp.utils.abs_squared(torch.fft.fft(psi_tilde))
        rho = (f @ psi_sqr) * dz
        return E, rho

    def compute(self, state, energy_only: bool) -> None:  # type: ignore
        ...

    def random_direction(self):
        ...

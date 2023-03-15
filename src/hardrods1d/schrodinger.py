import qimpy as qp
import numpy as np
import hardrods1d as hr
import torch
from scipy.special import expit
from IPython import embed


class SchrodingerFMT():  # type: ignore
    """Exact 1D Schrodinger solver"""

    grid1d: hr.Grid1D
    T: float  #: Temperature
    n_bulk: float  #: Bulk number density of the fluid
    mu: float  #: Bulk chemical potential
    e_bulk: float  #: Bulk energy density
    logn: qp.grid.FieldR  #: State of the classical DFT (effectively local mu)
    V: qp.grid.FieldR  #: External potential
    w0_tilde: torch.Tensor
    w1_tilde: torch.Tensor

    def __init__(self, grid1d: hr.Grid1D, *, n_bulk: float) -> None:
        """Initializes to bulk fluid with no external potential."""
        self.grid1d = grid1d
        self.T = 1.0 # Needs to be changed
        self.set_n_bulk(n_bulk)
        self.n = qp.grid.FieldR(
            grid1d.grid, data=torch.zeros_like(grid1d.z)
        )
        self.V = self.n.zeros_like()

    def set_n_bulk(self, n_bulk: float) -> None:
        """Update the chemical potential to target given bulk density."""
        self.n_bulk = n_bulk
        # Bulk free energy density, omega(n) = nT(log(n) - log(1-2Rn)) - mu n
        # Derivative domega/dn = T [log(n) - log(1-2Rn) + 1 + 2Rn/(1-2Rn)] - mu = 0
        self.mu = (np.pi*n_bulk)**2/8

    def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
        self.logn += step_size * direction

    def finite_difference_test(*args, **kwargs):
        ...

    def minimize(self):
        L = self.grid1d.L
        dz = self.grid1d.dz
        Nx = int(np.round(L / dz))
        def schrodinger_1d(k, n_bulk, T):
            # Full KE operator:
            iG = np.concatenate((np.arange(Nx//2), np.arange(Nx//2-Nx, 0)))
            k_plus_G = (iG + k) * (2*np.pi/L)
            KE_diag = 0.5*(k_plus_G**2)

            # Reduced KE for plane-wave basis:
            Gmax_rho = (np.pi/L) * Nx  # Nyquist frequency for charge density grid
            Gmax_wfn = 0.5 * Gmax_rho
            KEcut_wfn = 0.5 * (Gmax_wfn**2)
            sel = np.where(KE_diag <= KEcut_wfn)[0]
            iG = iG[sel]
            KE = np.diag(KE_diag[sel])
            n_bands = len(sel)

            # Potential operator in PW basis
            V = np.array(np.squeeze(self.V.data))
            Vtilde = np.fft.ifft(V)
            ij_grid = (iG[:, None] - iG[None, :]) % Nx
            Vop = Vtilde[ij_grid]
            H = KE + Vop

            # Diagonalize the Hamiltonian to find the eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(H)

            # Expand wavefunctions for density compute:
            psi_tilde = np.zeros((n_bands, Nx), dtype=np.complex128)
            psi_tilde[:, iG] = eigvecs.T

            # FD smearing
            def fd(E, n_bulk, T):
                return expit((self.mu-E)/T)

            psi_sqr = abs(np.fft.fft(psi_tilde))**2
            smearing_weights = fd(eigvals, n_bulk, T)
            rho = smearing_weights @ psi_sqr
            return np.average(eigvals, weights=smearing_weights), eigvecs, rho
        N_k_points = 100
        k_points = np.linspace(-np.pi/L,np.pi/L,N_k_points)
        k_dens = []
        k_energy = []
        for i, k in enumerate(k_points):
            energy, eigvecs, rho = schrodinger_1d(k, self.n_bulk, self.T)
            k_dens.append(rho)
            k_energy.append(energy)
        rho = np.array(k_dens).mean(axis=0) * 2
        self.n.data = torch.from_numpy(rho[None, None, :])
        return np.mean(k_energy)

    def compute(self, state, energy_only: bool) -> None:  # type: ignore
        ...

    def random_direction(self):
        ...

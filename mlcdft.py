import qimpy as qp
import numpy as np
import torch
from grid1d import Grid1D
from nn_function import NNFunction


class MLCDFT(torch.nn.Module):  # type: ignore
    """Machine-learned CDFT in 1D."""

    T: float  #: Temperature
    w: NNFunction  #: Weight functions defining spatial nonlocality
    f_ex: NNFunction  #: Free energy density as a function of weighted densities

    def __init__(
        self,
        *,
        T: float,
        w: NNFunction,
        f_ex: NNFunction,
    ) -> None:
        """Initializes to bulk fluid with no external potential."""
        super().__init__()
        self.T = T
        self.w = w
        self.f_ex = f_ex
        # Check dimensions:
        assert w.n_in == 1
        assert w.n_out == f_ex.n_in
        assert f_ex.n_out == w.n_out

    def get_energy(self, n: qp.grid.FieldR, V_minus_mu: qp.grid.FieldR) -> qp.Energy:
        w_tilde = self.w(MLCDFT.Gmag(n.grid)[None])
        n_bar = n[None, ...].convolve(w_tilde)

        energy = qp.Energy()
        energy["Omega0"] = n ^ (self.T * n.log() + V_minus_mu)  # Ideal-gas part
        energy["Fex"] = self.T * (n_bar ^ self.f_ex(n_bar)).sum()  # Excess part
        return energy

    def get_mu(self, n_bulk: float) -> float:
        """Compute chemical potential that will produce target density `n_bulk`."""
        # Bulk condition: (d/dn) [f_id(n) + f_ex(w * n)] = mu
        w_bulk = self.w(torch.zeros(1, device=qp.rc.device))
        n = torch.tensor(n_bulk)
        n.requires_grad = True
        n_bar = w_bulk * n
        (self.T * (n * n.log() + n_bar @ self.f_ex(n_bar))).backward()
        return n.grad.item()

    @staticmethod
    def Gmag(grid: qp.grid.Grid):
        iG = grid.get_mesh("H").to(torch.double)  # half-space
        return (iG @ grid.lattice.Gbasis.T).norm(dim=-1)


class MLCDFT_minimizer(qp.utils.Minimize[qp.grid.FieldR]):

    mlcdft: MLCDFT  #: Classical DFT free energy functional
    grid1d: Grid1D
    logn: qp.grid.FieldR  #: State of the classical DFT (effectively local mu)
    V: qp.grid.FieldR  #: External potential
    n_bulk: float  #: Bulk number density of the fluid
    mu: float  #: Bulk chemical potential

    def __init__(self, *, mlcdft: MLCDFT, grid1d: Grid1D, n_bulk: float) -> None:
        super().__init__(
            comm=qp.rc.comm,
            checkpoint_in=qp.utils.CpPath(),
            name="MLCDFT",
            n_iterations=1000,
            energy_threshold=1e-9,
            extra_thresholds={"|grad|": 1e-8},
            method="cg",
            n_consecutive=1,
        )
        self.mlcdft = mlcdft
        self.grid1d = grid1d
        self.n_bulk = n_bulk
        self.mu = mlcdft.get_mu(n_bulk)
        self.logn = qp.grid.FieldR(
            grid1d.grid, data=torch.full_like(grid1d.z, np.log(self.n_bulk))
        )
        self.V = self.logn.zeros_like()

    @property
    def n(self) -> qp.grid.FieldR:
        """Get current density (from self.logn)"""
        return self.logn.exp()

    @n.setter
    def n(self, new_n: qp.grid.FieldR) -> None:
        self.logn = new_n.log()

    def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
        self.logn += step_size * direction

    def compute(self, state, energy_only: bool) -> None:  # type: ignore
        if not energy_only:
            self.logn.data.requires_grad = True
            self.logn.data.grad = None

        n = self.n
        state.energy = self.mlcdft.get_energy(n, self.V - self.mu)

        # Gradient computation:
        if not energy_only:
            sum(state.energy.values()).backward()  # derivative -> self.logn.data.grad
            state.gradient = qp.grid.FieldR(n.grid, data=self.logn.data.grad)
            state.K_gradient = state.gradient
            state.extra = [state.gradient.norm()]
            self.logn.data.requires_grad = False

    def random_direction(self) -> qp.grid.FieldR:
        grid = self.grid1d.grid
        return qp.grid.FieldR(grid, data=torch.randn(grid.shapeR_mine))

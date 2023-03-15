import qimpy as qp
import numpy as np
import torch


class Grid1D:
    L: float
    dz: float
    lattice: qp.lattice.Lattice
    grid: qp.grid.Grid
    iGz: torch.Tensor
    Gmag: torch.Tensor
    z: torch.Tensor

    def __init__(self, *, L: float, dz: float, parallel: bool = True) -> None:
        """Create QimPy lattice and grid suitable for 1D calculations.
        The x and y directions are set to have a single grid point and length 1.
        The z direction is periodic with length `L` and nominal spacing `dz`.
        The grid is split over MPI if `parallel` = True.
        """
        self.L = L
        self.lattice = qp.lattice.Lattice(system="orthorhombic", a=1.0, b=1.0, c=L)
        ions = qp.ions.Ions(lattice=self.lattice)
        symmetries = qp.symmetries.Symmetries(
            lattice=self.lattice,
            ions=ions,
            axes={"V": np.array((0.0, 0.0, 1.0))},
            # z-axis symmetry broken by ext. potential
        )
        Nz = int(np.round(L / dz))  # based on nominal grid spacing
        self.dz = L / Nz
        self.grid = qp.grid.Grid(
            lattice=self.lattice,
            symmetries=symmetries,
            comm=(qp.rc.comm if parallel else None),
            shape=(1, 1, Nz),
        )
        iG = self.grid.get_mesh("H").to(torch.double)  # half-space
        self.iGz = iG[..., 2]
        self.Gmag = (iG @ self.lattice.Gbasis.T).norm(dim=-1)
        r = (
            self.grid.get_mesh("R") / torch.tensor(self.grid.shape, device=qp.rc.device)
        ) @ self.grid.lattice.Rbasis.T
        self.z = r[..., 2]


def get1D(x: torch.Tensor) -> np.ndarray:
    return x[0, 0].to(qp.rc.cpu).numpy()


def trapz(f: np.ndarray, h: float) -> np.ndarray:
    """Cumulative trapezoidal integral of a function `f` sampled at spacing `h`."""
    return np.concatenate(([0.0], np.cumsum(0.5 * (f[:-1] + f[1:])) * h))

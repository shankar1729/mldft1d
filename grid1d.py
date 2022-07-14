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

    def __init__(self, *, L: float, dz: float) -> None:
        """Create QimPy lattice and grid suitable for 1D calculations.
        The x and y directions are set to have a single grid point and length 1.
        The z direction is periodic with length `L` and nominal spacing `dz`.
        """
        self.L = L
        self.lattice = qp.lattice.Lattice(system="orthorhombic", a=1., b=1., c=L)
        process_grid = qp.utils.ProcessGrid(qp.rc.comm, "rkb", (1, 1, -1))
        ions = qp.ions.Ions(lattice=self.lattice, process_grid=process_grid)
        symmetries = qp.symmetries.Symmetries(
            lattice=self.lattice,
            ions=ions,
            axes={'V': np.array((0., 0., 1.))},
            # z-axis symmetry broken by ext. potential
        )
        Nz = int(np.round(L / dz))  # based on nominal grid spacing
        self.dz = L / Nz
        self.grid = qp.grid.Grid(
            lattice=self.lattice,
            symmetries=symmetries,
            comm=qp.rc.comm,
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


# --- Super hacky stuff that needs to be merged into QimPy ---

def fieldR_log(self: qp.grid.FieldR) -> qp.grid.FieldR:
    return qp.grid.FieldR(self.grid, data=self.data.log())


def fieldR_exp(self: qp.grid.FieldR) -> qp.grid.FieldR:
    return qp.grid.FieldR(self.grid, data=self.data.exp())


def scalar_rsub(self: qp.grid.FieldR, a: float) -> qp.grid.FieldR:
    return qp.grid.FieldR(self.grid, data=(a - self.data))


def scalar_radd(self: qp.grid.FieldR, a: float) -> qp.grid.FieldR:
    return qp.grid.FieldR(self.grid, data=(a + self.data))


def fieldR_div(self: qp.grid.FieldR, other: qp.grid.FieldR) -> qp.grid.FieldR:
    return qp.grid.FieldR(self.grid, data=self.data / other.data)


def fieldR_convolve(self: qp.grid.FieldR, kernel_tilde: torch.Tensor) -> qp.grid.FieldR:
    return ~qp.grid.FieldH(self.grid, data=kernel_tilde * (~self).data)


qp.grid.FieldR.log = fieldR_log
qp.grid.FieldR.exp = fieldR_exp
qp.grid.FieldR.__rsub__ = scalar_rsub
qp.grid.FieldR.__radd__ = scalar_radd
qp.grid.FieldR.__truediv__ = fieldR_div
qp.grid.FieldR.convolve = fieldR_convolve

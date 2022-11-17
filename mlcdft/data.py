import qimpy as qp
from h5py import File
from ..grid1d import Grid1D


class Data:
    grid1d: Grid1D
    V: torch.Tensor
    n: torch.Tensor
    E: torch.Tensor
    n_bulk: float
    T: float
    R: float

    def __init__(self, filename: str) -> None:
        """Load ML-CDFT training data from HDF5 file `filename`."""
        f = File(filename, "r")
        z = torch.tensor(f["z"], device=qp.rc.device)
        self.V = torch.tensor(f["V"], device=qp.rc.device)
        self.n = torch.tensor(f["n"], device=qp.rc.device)
        self.E = torch.tensor(f["E"], device=qp.rc.device)
        self.n_bulk = float(f.attrs["n_bulk"])
        self.T = float(f.attrs["T"])
        self.R = float(f.attrs["R"])

        # Create grid:
        dz = (z[1] - z[0]).item()
        L = (z[-1] - z[0]).item() + dz
        self.grid1d = Grid1D(L=L, dz=dz)
        assert len(z) == len(self.grid1d.z)

import sys
import torch
import qimpy as qp
import numpy as np
import hardrods1d as hr
from h5py import File


class Data:
    grid1d: hr.Grid1D
    n_perturbations: int
    E: torch.Tensor
    n_bulk: float
    T: float
    R: float
    V: qp.grid.FieldR
    n: qp.grid.FieldR

    def __init__(self, filename: str) -> None:
        """Load ML-CDFT training data from HDF5 file `filename`."""
        f = File(filename, "r")
        z = torch.tensor(f["z"], device=qp.rc.device)
        lbda = torch.tensor(f["lbda"], device=qp.rc.device)
        V = torch.outer(lbda, torch.tensor(f["V"], device=qp.rc.device))
        n = torch.tensor(np.array(f["n"]), device=qp.rc.device)
        self.n_perturbations = len(lbda)
        self.E = torch.tensor(f["E"], device=qp.rc.device)
        self.n_bulk = float(f.attrs["n_bulk"])
        self.T = float(f.attrs["T"])
        self.R = float(f.attrs["R"])

        # Create grid:
        dz = (z[1] - z[0]).item()
        L = (z[-1] - z[0]).item() + dz
        self.grid1d = hr.Grid1D(L=L, dz=dz)
        assert len(z) == self.grid1d.z.shape[2]

        # Create fieldR's for n and V:
        self.V = qp.grid.FieldR(self.grid1d.grid, data=V[:, None, None, :])
        self.n = qp.grid.FieldR(self.grid1d.grid, data=n[:, None, None, :])

    def __repr__(self) -> str:
        return (
            "hardrods1d.mlcdft.Data("
            f"T={self.T}, R={self.R}, n_bulk={self.n_bulk}, L={self.grid1d.L}, "
            f"n_perturbations={self.n_perturbations}"
            ")"
        )


def main() -> None:
    for filename in sys.argv[1:]:
        print(Data(filename))


if __name__ == "__main__":
    main()

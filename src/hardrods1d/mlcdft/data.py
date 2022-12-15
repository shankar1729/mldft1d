import sys
import copy
import torch
import functools
import qimpy as qp
import numpy as np
import hardrods1d as hr
from h5py import File
from typing import Sequence, TypeVar, List, Optional


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
        self.grid1d = get_grid1d(L, dz)
        assert len(z) == self.grid1d.z.shape[2]

        # Create fieldR's for n and V:
        self.V = qp.grid.FieldR(self.grid1d.grid, data=V[:, None, None, :])
        self.n = qp.grid.FieldR(self.grid1d.grid, data=n[:, None, None, :])

    def __repr__(self) -> str:
        return (
            "hardrods1d.mlcdft.Data("
            f"T={self.T}, R={self.R}, n_bulk={self.n_bulk}, L={self.grid1d.L:.2f}, "
            f"n_perturbations={self.n_perturbations}"
            ")"
        )


def fuse_data(data_arr: Sequence[Data]) -> List[Data]:
    """Fuse entries with same grid and n_bulk into concatenated entries."""
    result = []
    remainder = data_arr
    while remainder:
        # Find entries with same grid and n_bulk as first one:
        ref_grid1d = remainder[0].grid1d
        ref_n_bulk = remainder[0].n_bulk
        same_grid: List[Data] = []
        next_remainder: List[Data] = []
        for data in remainder:
            (
                same_grid
                if ((data.grid1d is ref_grid1d) and (data.n_bulk == ref_n_bulk))
                else next_remainder
            ).append(data)
        remainder = next_remainder

        # Combine those entries into a single data set:
        combined = copy.copy(same_grid[0])
        combined.n_perturbations = sum(data.n_perturbations for data in same_grid)
        combined.E = torch.cat([data.E for data in same_grid])
        combined.V = qp.grid.FieldR(
            ref_grid1d.grid, data=torch.cat([data.V.data for data in same_grid], dim=0)
        )
        combined.n = qp.grid.FieldR(
            ref_grid1d.grid, data=torch.cat([data.n.data for data in same_grid], dim=0)
        )
        result.append(combined)
    return result


def main() -> None:
    for filename in sys.argv[1:]:
        print(Data(filename))


T = TypeVar("T")


def random_split(
    data: Sequence[T], counts: Sequence[int], seed: Optional[int] = None
) -> Sequence[Sequence[T]]:
    generator = None if (seed is None) else torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(data, counts, generator)  # type: ignore


def random_batch_split(
    data: Sequence[T], batch_size: int, seed: Optional[int] = None
) -> Sequence[Sequence[T]]:
    # Determine even batch sizes:
    n_data = len(data)
    n_batches = qp.utils.ceildiv(n_data, batch_size)
    i_batch = np.arange(n_batches, dtype=int)
    batch_start = (i_batch * n_data) // n_batches
    batch_stop = ((i_batch + 1) * n_data) // n_batches
    counts = batch_stop - batch_start
    return random_split(data, counts, seed)


@functools.lru_cache()
def get_grid1d(L: float, dz: float) -> hr.Grid1D:
    """Get/make a 1D grid of length `L` and spacing `dz` (cached by L, dz)."""
    qp.log.info(f"\n----- Making 1D grid for {L = :.2f} and {dz = :.3f} -----")
    return hr.Grid1D(L=L, dz=dz, parallel=False)  # use data-parallel instead


if __name__ == "__main__":
    main()

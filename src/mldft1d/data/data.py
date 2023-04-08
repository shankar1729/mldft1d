import sys
import copy
import torch
import functools
import qimpy as qp
import numpy as np
from mldft1d import Grid1D
from mpi4py import MPI
from h5py import File
from typing import Sequence, TypeVar, List, Optional


class Data:
    grid1d: Grid1D
    n_perturbations: int
    E: torch.Tensor
    V_minus_mu: qp.grid.FieldR
    n: qp.grid.FieldR
    attrs: dict[str, float]

    def __init__(self, filename: str) -> None:
        """Load ML-CDFT training data from HDF5 file `filename`."""
        f = File(filename, "r")
        z = torch.tensor(f["z"], device=qp.rc.device)
        lbda = torch.tensor(f["lbda"], device=qp.rc.device)
        mu = float(f.attrs["mu"])
        V_minus_mu = torch.outer(lbda, torch.tensor(f["V"], device=qp.rc.device)) - mu
        if "V0" in f:
            V_minus_mu += torch.tensor(f["V0"], device=qp.rc.device)
        n = torch.tensor(np.array(f["n"]), device=qp.rc.device)
        self.n_perturbations = len(lbda)
        self.E = torch.tensor(f["E"], device=qp.rc.device)

        # Read scalar attributes:
        self.attrs = dict()
        for key in f.attrs:
            if key not in {"n_bulk", "mu"}:  # n_bulk and mu captured in V - mu
                self.attrs[key] = float(f.attrs[key])

        # Create grid:
        dz = (z[1] - z[0]).item()
        L = (z[-1] - z[0]).item() + dz
        self.grid1d = get_grid1d(L, dz)
        assert len(z) == self.grid1d.z.shape[2]

        # Create fieldR's for n and V:
        self.V_minus_mu = qp.grid.FieldR(
            self.grid1d.grid, data=V_minus_mu[:, None, None, :]
        )
        self.n = qp.grid.FieldR(self.grid1d.grid, data=n[:, None, None, :])

    def __repr__(self) -> str:
        attrs = self.attrs
        L = self.grid1d.L
        n_perturbations = self.n_perturbations
        return f"hardrods1d.Data({L=:.2f}, {n_perturbations=}, {attrs=})"


def fuse_data(data_arr: Sequence[Data]) -> List[Data]:
    """Fuse entries with same grid and attrs into concatenated entries."""
    result = []
    remainder = data_arr
    while remainder:
        # Find entries with same grid and n_bulk as first one:
        ref_grid1d = remainder[0].grid1d
        ref_attrs = remainder[0].attrs
        same_grid: List[Data] = []
        next_remainder: List[Data] = []
        for data in remainder:
            (
                same_grid
                if ((data.grid1d is ref_grid1d) and (data.attrs == ref_attrs))
                else next_remainder
            ).append(data)
        remainder = next_remainder

        # Combine those entries into a single data set:
        combined = copy.copy(same_grid[0])
        combined.n_perturbations = sum(data.n_perturbations for data in same_grid)
        combined.E = torch.cat([data.E for data in same_grid])
        combined.V_minus_mu = qp.grid.FieldR(
            ref_grid1d.grid,
            data=torch.cat([data.V_minus_mu.data for data in same_grid], dim=0),
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
    data: Sequence[T], n_batches: int, seed: Optional[int] = None
) -> Sequence[Sequence[T]]:
    # Determine even batch sizes:
    n_data = len(data)
    i_batch = np.arange(n_batches, dtype=int)
    batch_start = (i_batch * n_data) // n_batches
    batch_stop = ((i_batch + 1) * n_data) // n_batches
    counts = batch_stop - batch_start
    return random_split(data, counts, seed)  # type: ignore


def random_mpi_split(data: Sequence[T], comm: MPI.Comm, seed: int = 0) -> Sequence[T]:
    """Randomly split sequence over `comm` and return portion on this process.
    Seed must be used to synchronize splits over the processes"""
    return random_batch_split(data, comm.size, seed)[comm.rank]


@functools.lru_cache()
def get_grid1d(L: float, dz: float) -> Grid1D:
    """Get/make a 1D grid of length `L` and spacing `dz` (cached by L, dz)."""
    qp.log.info(f"\n----- Making 1D grid for {L = :.2f} and {dz = :.3f} -----")
    return Grid1D(L=L, dz=dz, parallel=False)  # use data-parallel instead


if __name__ == "__main__":
    main()

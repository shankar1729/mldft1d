from __future__ import annotations
from typing import Sequence, TypeVar, List, Optional
import functools
import sys
import copy

import torch
import numpy as np
from mpi4py import MPI
from h5py import File

from qimpy import rc, log
from qimpy.grid import FieldR
from mldft1d import Grid1D


class Data:
    grid1d: Grid1D
    n_perturbations: int
    E: torch.Tensor
    n: FieldR  #: stored as n_sites x n_batch x grid-dimensions
    dE_dn: FieldR
    attrs: dict[str, float]

    def __init__(self, filename: str) -> None:
        """Load ML-CDFT training data from HDF5 file `filename`."""
        f = File(filename, "r")
        z = torch.tensor(f["z"], device=rc.device)
        self.E = torch.tensor(f["E"], device=rc.device)
        self.n_perturbations = len(self.E)
        n = torch.tensor(np.array(f["n"]), device=rc.device).swapaxes(0, 1)
        dE_dn = torch.tensor(np.array(f["dE_dn"]), device=rc.device).swapaxes(0, 1)

        # Read scalar attributes:
        self.attrs = dict()
        for key in f.attrs:
            value = f.attrs[key]
            if hasattr(value, "__len__"):
                # Flatten-out vector attributes in data files into scalar ones:
                for index, value_i in enumerate(value):
                    self.attrs[f"{key}_{index}"] = float(value_i)
            else:
                self.attrs[key] = float(value)

        # Create grid:
        dz = (z[1] - z[0]).item()
        L = (z[-1] - z[0]).item() + dz
        self.grid1d = get_grid1d(L, dz)
        assert len(z) == self.grid1d.z.shape[2]

        # Create fieldR's for n and dE_dn:
        self.n = FieldR(self.grid1d.grid, data=n[..., None, None, :])
        self.dE_dn = FieldR(self.grid1d.grid, data=dE_dn[..., None, None, :])

    def __repr__(self) -> str:
        attrs = self.attrs
        L = self.grid1d.L
        n_perturbations = self.n_perturbations
        return f"mldft1d.data.Data({L=:.2f}, {n_perturbations=}, {attrs=})"


def fuse_data(data_arr: Sequence[Data], batch_size: int) -> List[Data]:
    """Fuse entries with same grid and attrs into concatenated entries."""
    result = []
    remainder = data_arr
    while remainder:
        # Find entries with same grid as first one:
        ref_grid1d = remainder[0].grid1d
        ref_attrs = remainder[0].attrs
        same_grid: List[Data] = []
        next_remainder: List[Data] = []
        n_perts = 0
        has_room = True
        for data in remainder:
            if has_room and (data.grid1d is ref_grid1d) and (data.attrs == ref_attrs):
                same_grid.append(data)
                n_perts += data.n_perturbations
                if n_perts > batch_size:
                    has_room = False  # batch size reached, don't fuse any more
            else:
                next_remainder.append(data)
        remainder = next_remainder

        # Combine those entries into a single data set:
        combined = copy.copy(same_grid[0])
        combined.n_perturbations = sum(data.n_perturbations for data in same_grid)
        combined.E = torch.cat([data.E for data in same_grid])
        combined.n = FieldR(
            ref_grid1d.grid, data=torch.cat([data.n.data for data in same_grid], dim=1)
        )
        combined.dE_dn = FieldR(
            ref_grid1d.grid,
            data=torch.cat([data.dE_dn.data for data in same_grid], dim=1),
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
    log.info(f"\n----- Making 1D grid for {L = :.2f} and {dz = :.3f} -----")
    return Grid1D(L=L, dz=dz, parallel=False)  # use data-parallel instead


if __name__ == "__main__":
    main()

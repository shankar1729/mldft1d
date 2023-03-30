from .. import Grid1D, Minimizer, protocols, nn
from . import Schrodinger, ThomasFermi
import qimpy as qp


def exact(*, grid1d: Grid1D, n_bulk: float, T: float) -> protocols.DFT:
    """Make exact Kohn-Sham DFT (Schrodinger solver)."""
    return Schrodinger(grid1d, n_bulk=n_bulk, T=T)


def ml(*, grid1d: Grid1D, n_bulk: float, load_file: str) -> protocols.DFT:
    """Make approximate Kohn-Sham DFT using an ML kinetic energy functional."""
    return Minimizer(
        functionals=[
            nn.Functional.load(qp.rc.comm, load_file=load_file, f_bulk=ThomasFermi())
        ],
        grid1d=grid1d,
        n_bulk=n_bulk,
        name="MLEDFT",
    )

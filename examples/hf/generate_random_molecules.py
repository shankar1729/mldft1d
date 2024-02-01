import qimpy as qp
import numpy as np
from mldft1d.data.generate import run
from scipy.stats import randint, loguniform


qp.io.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

n_batch = 1
prefix = "random_molecules/molecule"
dz = 0.1  # grid resolution
padding = 20.0  # distance between periodic images

comm = qp.rc.comm
division = qp.mpi.TaskDivision(n_tot=n_batch, n_procs=comm.size, i_proc=comm.rank)
for i_batch in range(1 + division.i_start, 1 + division.i_stop):
    qp.log.warning(
        f"\n---- Generating {i_batch} of {n_batch} on process {comm.rank} ----\n"
    )

    # Generate molecule parameters (custom sampling since highly constrained)
    n_atoms = randint(1, 4).rvs()
    Z = randint(1, 4).rvs(size=n_atoms)
    Z_tot = Z.sum()
    bond_lengths = loguniform(0.5, 5.0).rvs(size=n_atoms)
    positions = np.cumsum(bond_lengths)

    # Center in box with required padding:
    pos_min, pos_max = positions[[0, -1]]  # first and last atom positions
    L = np.ceil((pos_max - pos_min + padding) / dz) * dz
    positions += 0.5 * L - 0.5 * (pos_min + pos_max)

    try:
        run(
            filename=f"{prefix}{i_batch}.h5",
            functional="hf",
            dz=dz,
            L=L,
            n_bulk=(Z_tot / L),
            T=0.01,  # Fermi smearing width
            lbda=dict(min=1.0, max=1.1, step=1.0),  # scale = 1 only (i.e. no scaling)
            periodic=True,
            Vshape=dict(shape="coulomb1d", ionpos=positions, Zs=-Z, fractional=False),
        )
    except Exception as err:
        qp.log.error(f"Failed with exception {err}")

qp.rc.report_end()
qp.profiler.StopWatch.print_stats()

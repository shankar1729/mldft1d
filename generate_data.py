import qimpy as qp
import numpy as np
from grid1d import Grid1D, get1D
from hard_rods_fmt import HardRodsFMT
import h5py

qp.utils.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

def run(*,
        L: float,
        dz: float,
        R: float,  #: Radius / half-length `R`
        T: float, #: Temperature
        n_bulk: float, #: Bulk number density of the fluid
        Vsigma: float,
        lbda: dict, #: min: float, max: float, step: float, 
        filename: str #: hdf5 filename, must end with .hdf5
        ):
        
    grid1d = Grid1D(L=L, dz=dz)
    cdft = HardRodsFMT(grid1d, R=R, T=T, n_bulk=n_bulk)
    
    qp.log.info(f"mu = {cdft.mu}")
    Vshape = qp.grid.FieldR(grid1d.grid, data=(-0.5 * ((grid1d.z - 0.5 * grid1d.L) / Vsigma).square()).exp())

    n = []
    E = []
    P_lambda = np.arange(start = lbda['min'], stop = lbda['max'], step = lbda['step'], dtype = 'f')
    for V0 in P_lambda:
        # cdft.V.data = (0.5 * V0) * ((0.4 * grid1d.L - (grid1d.z - 0.5 * grid1d.L).abs()) / Vsigma).erfc()
        cdft.V = V0 * Vshape
        cdft.finite_difference_test(cdft.random_direction())
        E.append(cdft.minimize())
        n.append(get1D(cdft.n.data))


    f = h5py.File(filename, "w")

    f['z'] = get1D(grid1d.z)
    f['V'] = get1D(Vshape.data)
    f['lambda'] = P_lambda
    f['n'] = np.array(n)

    dset5 = f.create_dataset("E", (1,len(E)), dtype = 'f')
    dset5[...] = E

    f.attrs["n_bulk"] = 0.6 

    f.close()
    return 

import yaml

with open(r'Params.yaml') as file:
    Params = yaml.load(file, Loader=yaml.FullLoader)

run(**Params)


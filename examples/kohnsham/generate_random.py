import qimpy as qp
import numpy as np
from mldft1d.data.generate import batch, Choice
from scipy.stats import lognorm, randint, loguniform

qp.utils.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

batch(
    n_batch=1000,
    prefix="random_data/random",
    functional="kohnsham",
    L=Choice(np.arange(4.0, 20.0, 0.2)),
    dz=0.05,
    T=0.01,  # Fermi smearing width
    n_bulk=lognorm(s=0.5, scale=0.5),
    Vshape=dict(
        shape="random",
        sigma=loguniform(a=0.1, b=3.0),
        seed=randint(1, np.iinfo(np.int32).max),
    ),
    lbda=dict(min=0.0, max=5.0, step=0.2),
)

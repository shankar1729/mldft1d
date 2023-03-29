import qimpy as qp
import numpy as np
from mldft1d.data.generate import batch, Choice
from scipy.stats import lognorm, randint, uniform

qp.utils.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

batch(
    n_batch=10,
    prefix="random_data/random",
    functional="kohnsham",
    L=Choice(np.arange(2.0, 10.0, 0.1)),
    dz=0.05,
    T=0.01,  # Fermi smearing width
    n_bulk=lognorm(s=0.5, scale=0.5),
    Vshape=dict(
        shape="random",
        sigma=uniform(loc=0.5, scale=2.0),
        seed=randint(1, np.iinfo(np.int32).max),
    ),
    lbda=dict(min=0.0, max=1.0, step=0.1),
)

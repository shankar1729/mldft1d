import qimpy as qp
import numpy as np
from mldft1d.data.generate import batch, Choice
from scipy.stats import randint, uniform, loguniform

qp.io.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

batch(
    n_batch=1000,
    prefix="random_data/random",
    functional="ising",
    L=Choice(np.arange(40.0, 200.0, 2.0)),
    dz=1.0,
    T=Choice(np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])),
    J=-1.0,
    n_bulk=uniform(loc=0.1, scale=0.8),
    Vshape=dict(
        shape="random",
        sigma=loguniform(a=2.0, b=10.0),
        seed=randint(1, np.iinfo(np.int32).max),
    ),
    lbda=dict(min=0.0, max=2.0, step=0.1),
)

import qimpy as qp
import numpy as np
from mldft1d.data.generate import batch, Choice
from scipy.stats import randint, uniform

qp.io.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

batch(
    n_batch=1000,
    prefix="random_data/random",
    functional="hardrods",
    L=Choice(np.arange(4.0, 20.0, 0.2)),
    dz=0.05,
    R=0.5,  # Hard rod radius (half-length)
    T=1.0,  # Temperature (overall scale in hard rod KE and entropy)
    n_bulk=uniform(loc=0.1, scale=0.7),  # Note that n_bulk must be < 1
    Vshape=dict(
        shape="random",
        sigma=uniform(loc=0.1, scale=2.0),
        seed=randint(1, np.iinfo(np.int32).max),
    ),
    lbda=dict(min=0.0, max=5.0, step=0.5),  # Make sure some perturbing V >> T
)

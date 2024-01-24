import qimpy as qp
import numpy as np
from mldft1d.data.generate import batch, Choice
from scipy.stats import randint, uniform

qp.io.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

# Create n_bulk pairs for mixture (constrained by packing limit)
R = [0.4, 0.6]  # Hard rod radius (half-length)
n_bulk_choices = []
for i_repeat in range(1000):
    # Create a pair with sum(2R * n) <= max_pack_fraction
    max_pack_fraction = 0.7
    n_bulk_0 = np.random.rand() * max_pack_fraction / (2 * R[0])
    max_pack_fraction -= n_bulk_0 * (2 * R[0])
    n_bulk_1 = np.random.rand() * max_pack_fraction / (2 * R[1])
    n_bulk_choices.append((n_bulk_0, n_bulk_1))

batch(
    n_batch=1,
    prefix="random_data/random",
    functional="hardrods",
    L=Choice(np.arange(4.0, 20.0, 0.2)),
    dz=0.05,
    R=R,  # Hard rod radius (half-length)
    T=1.0,  # Temperature (overall scale in hard rod KE and entropy)
    n_bulk=Choice(n_bulk_choices),
    Vshape=[
        dict(
            shape="random",
            sigma=uniform(loc=0.2, scale=2.0),
            seed=randint(1, np.iinfo(np.int32).max),
        )
    ]
    * 2,
    lbda=dict(min=0.0, max=5.0, step=0.5),  # Make sure some perturbing V >> T
)

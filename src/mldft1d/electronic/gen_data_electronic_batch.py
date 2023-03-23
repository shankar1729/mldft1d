import os
import numpy as np
from gen_data_electronic import main

n_bulks = np.array([0.4, 0.6, 0.8])
Ls = np.array([2, 5, 10])
sigmas = np.array([0.2, 0.5, 1.0, 2.0])
seed = 0
for n_bulk in n_bulks:
    for L in Ls:
        for sigma in sigmas[sigmas <= 0.5 * L]:  # avoid wasted flat potentials
            for i_seed in range(10):
                seed += 1
                os.environ["n_bulk"] = str(n_bulk)
                os.environ["L"] = str(L)
                os.environ["sigma"] = str(sigma)
                os.environ["seed"] = str(seed)
                main()

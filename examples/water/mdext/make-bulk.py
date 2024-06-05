import numpy as np
import h5py

n, a, a_n = np.loadtxt("a_ex.dat").T
sel = np.where(np.logical_and(n >= 0.01, n <= 0.064))[0]
np.random.shuffle(sel)  # randomize batching

# Divide into batches
batch_size = 10
n_sel = (len(sel) // batch_size) * batch_size
sel = sel[:n_sel].reshape(-1, batch_size)

# Make into a box with some length
L = 40  # to weight it similarly to bulk calcs
dz = 0.1
Nz = int(round(L / dz))
z = np.arange(Nz) * dz
V = np.zeros_like(z)
ones = np.ones(Nz)

for i_batch, sel_i in enumerate(sel):
    ni = np.outer(n[sel_i], ones)
    dE_dn_i = np.outer(a_n[sel_i], ones)
    Ei = a[sel_i] * L
    out_file = f"Bulk/bulk{i_batch}.h5"
    with h5py.File(out_file, "w") as fp:
        fp["z"] = z
        fp["V"] = V[None]
        fp["n"] = ni[:, None]
        fp["E"] = Ei
        fp["dE_dn"] = dE_dn_i[:, None]
    print(f"Wrote {out_file}.")

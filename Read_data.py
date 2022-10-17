import h5py
import matplotlib.pyplot as plt
import numpy as np

f = h5py.File('data.hdf5', 'r')

z = f['z1d']
V = f['V']
P_lambda = f['lambda']
n = f['n']
E = f['E']

plt.plot(z[0], V[0], label="V")
plt.plot(z[0], n[5], label="n")
plt.axhline(0.6, color='k', ls='dotted')
plt.xlabel("z")
plt.legend()
plt.show()
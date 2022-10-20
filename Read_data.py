import h5py
import matplotlib.pyplot as plt
import numpy as np

f = h5py.File('data.hdf5', 'r')

z = f['z']
V = f['V']
P_lambda = f['lambda']
n = f['n']
E = f['E']


print(z.shape)
print(V.shape)
print(P_lambda.shape)
print(n.shape)
print(E.shape)

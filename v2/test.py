import numpy as np

nx = 4
ny = 4
nz = 4

X = np.arange(nx)
Y = np.arange(ny)
Z = np.arange(nz)

x,y,z = np.zeros((nx,ny, nz)),np.zeros((nx,ny, nz)), np.zeros((nx,ny,nz))
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            x[i,j,k] = X[i]
            y[i,j,k] = Y[j]
            z[i, j, k] = Z[k]

x = np.ravel(x, order='F')
print x

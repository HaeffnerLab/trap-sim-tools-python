'''
Functions for performing the spherical
harmonic expansion of the potential
'''
import numpy as np
import math as mt
from scipy.special import lpmv
from scipy import linalg, matrix
import scipy
#import pyshtools

def legendre(n,X):
    '''
    like the matlab function, returns an array of all the assosciated legendre functions of degree n
    and order m = 0,1.... n for each element in X
    '''
    r = []
    for m in range(n+1):
        r.append(lpmv(m,n,X))
    return r

def spher_harm_basis(r0, X, Y, Z, order):
    '''
    Computes spherical harmonics, just re-written matlab code
   
    Returns: Yxx, rnorm
    where Yxx is a 1D array of the spherical harmonic evaluated on the grid
    rnorm is a normalization factor for the spherical harmonics

    The function returns the coefficients in the order:[C00 C10 C11c C11s ]'
    These correspond to the multipoles in cartesian coordinares: 
    [c z -x -y (z^2-x^2/2-y^2/2) -3zx -3yz 3x^2-3y^2 6xy ... 15(3xy^2-x^3) 15(y^3-3yx^2)         ...]
     1 2  3  4       5             6    7     8       9  ...       15           16          17   ...

    Q(10)  0.5[2z^3-3(x^2+y^2)z]
    Q(11) -1.5[4xz^2-x(x^2+y^2)]
    Q(12) -1.5[4yz^2-(x^2+y^2)y]
    Q(13)  15[z(x^2-y^2)]
    Q(14)  30xyz
    Q(15)  15(3xy^2-x^3)
    Q(16)  15(y^3-3x^2y)

    Or in terms of the Littich thesis:
    M1 M3 M4 M2 M7 M8 M6 M9 M5 (Using the convention in G. Littich's master thesis (2011))
    0  1  2  3  4  5  6  7  8  (the ith component of Q matrix)
    '''

    #initialize grid with expansion point (r0) at 0
    x0,y0,z0 = r0
    
    nx = len(X)
    ny = len(Y)
    nz = len(Z)
    npts = nx*ny*nz

    y, x, z = np.meshgrid(Y-y0,X-x0,Z-z0)
    x, y, z = np.reshape(x,npts), np.reshape(y,npts), np.reshape(z,npts)

    #change variables
    r = np.sqrt(x*x+y*y+z*z)
    r_trans = np.sqrt(x*x+y*y)
    theta = np.arctan2(r_trans,z)
    phi = np.arctan2(y,x)

    # for now normalizing as in matlab code
    dl = Z[1]-Z[0]
    scale = 1#np.sqrt(np.amax(r)*dl)
    rs = r/(scale)

    Q = []
    Q.append(np.ones(npts))

    #real part of spherical harmonics
    for n in range(1,order+1):
        p = legendre(n,np.cos(theta))
        c = (rs**n)*p[0]
        Q.append(c)
        for m in range(1,n+1):
            c = (rs**n)*p[m]*np.cos(m*phi)
            Q.append(c)
            cn = (rs**n)*p[m]*np.sin(m*phi)
            Q.append(cn)

    Q = np.transpose(Q)

    return Q, scale

def spher_harm_expansion(potential_grid, r0, X, Y, Z, order):
    '''
    Compute the least-squares solution for the spherical harmonic expansion on potential_grid.
    Arguments:
    potential_grid: 3D array of potential values
    r0: list [x0, y0, z0] of the expansion point
    X, Y, Z: axis ranges for the potential grid
    order: int, order of the expansion
    '''
    # Convert the 3D DC potential into 1D array.
    # Numerically invert, here the actual expansion takes place and we obtain the expansion coefficients M_{j}.

    nx = len(X)
    ny = len(Y)
    nz = len(Z)
    npts = nx*ny*nz

    W=np.reshape(potential_grid,npts) # 1D array of all potential values
    W=np.array([W]).T # make into column array

    Yj, scale = spher_harm_basis(r0,X,Y,Z,order)
    #Yj, rnorm = spher_harm_basis_v2(r0, X, Y, Z, order)

    Mj=np.linalg.lstsq(Yj,W,rcond=None) 
    Mj=Mj[0] # array of coefficients

    # rescale to original units
    i = 0
    for n in range(1,order+1):
        for m in range(1,2*n+2):
            i += 1
            Mj[i] = Mj[i]/(scale**n)
    return Mj,Yj,scale

def spher_harm_cmp(Mj,Yj,scale,order):
    '''
    regenerates the potential (V) from the spherical harmonic coefficients. 
    '''
    V = []
    #unnormalize
    i=0
    for n in range(1,order+1):
        for m in range(1,2*n+2):
            i += 1
            Mj[i] = Mj[i]*(scale**n)
    W = np.dot(Yj,Mj)
    return np.real(W)

def nullspace(A,eps=1e-15):
    u,s,vh = np.linalg.svd(A)
    nnz = (s >= eps).sum()
    null_space = vh[nnz:].conj().T
    return null_space





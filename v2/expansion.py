'''
Functions for performing the spherical
harmonic expansion of the potential
'''
import numpy as np
import math as mt
from scipy.special import lpmv

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
   
    Returns: [Y00,Y-11,Y01,Y11,Y-22,Y-12,Y02,Y12,Y22...], rnorm
    where Yxx is a 1D array of the spherical harmonic evaluated on the grid
    rnorm is a normalization factor for the spherical harmonics
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
    dl = X[1]-X[0]
    scale = np.sqrt(np.amax(r)*dl)
    rs = r/scale

    Q = []
    Q.append(np.ones(len(x)))

    #real part of spherical harmonics
    for n in range(1,order+1):
        p = legendre(n,np.cos(theta))
        c = (rs**n)*p[0]
        Q.append(c)
        for m in range(1,n+1):
            c = (rs**n)*p[m]*np.cos((m)*phi)
            Q.append(c)
            cn = (rs**n)*p[m]*np.sin((m)*phi)
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

    Yj, rnorm = spher_harm_basis(r0,X,Y,Z,order)
    #Yj, rnorm = spher_harm_basis_v2(r0, X, Y, Z, order)

    Mj=np.linalg.lstsq(Yj,W,rcond=None) 
    Mj=Mj[0] # array of coefficients

    # rescale to original units
    i = 0
    for n in range(1,order+1):
        for m in range(1,2*n+2):
            i += 1
            Mj[i] = Mj[i]/(rnorm**n)
    return Mj

def compute_gradient(potential,nx,ny,nz):
    # computes gradient & hessian of potential

    hessian = np.empty((3,3,nx,ny,nz))

    grad = np.gradient(potential)

    for k, grad_k in enumerate(grad):
        grad2 = np.gradient(grad_k)
        for l, grad_kl in enumerate(grad2):
            hessian[k,l,:,:,:] = grad_kl

    return grad,hessian

def compute_multipoles(grad,hessian):
    # computes multipole contribution at each point.
    # follows convention in gapless

    multipoles = []
    multipoles.append(-1*grad[0])
    multipoles.append(-1*grad[1])
    multipoles.append(-1*grad[2])
    multipoles.append(0.5*0.25*(hessian[0][0] - hessian[1][1]))                   #xx - yy
    multipoles.append(0.5*0.25*(2*hessian[2][2] - hessian[0][0] - hessian[1][1]))  #zz-xx-yy
    multipoles.append(0.25*hessian[0][1])                                        #xy
    multipoles.append(0.25*hessian[1][2])                                        #zy
    multipoles.append(0.25*hessian[0][2])                                        #xz

    return multipoles




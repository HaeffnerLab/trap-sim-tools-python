'''
Functions for performing the spherical
harmonic expansion of the potential
'''
import numpy as np
import math as mt
from scipy.special import lpmv, sph_harm


def spher_harm_basis(r0, X, Y, Z, order):
    """
    Computes real spherical harmonics on a flattened grid
    about expansion point r0 = [x0, y0, z0].

    Returns: [Y00,Y-11,Y01,Y11,Y-22,Y-12,Y02,Y12,Y22...], rnorm
    where Yxx is a 1D array of the spherical harmonic evaluated on the grid
    rnorm is a normalization factor for the spherical harmonics
     """

    # Construct variables from axes; no meshgrid as of 6/4/14; no potential as of 6/12/14
    nx,ny,nz=X.shape[0],Y.shape[0],Z.shape[0]
    x,y,z = np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
    x0, y0, z0 = r0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x[i,j,k] = X[i] - x0
                y[i,j,k] = Y[j] - y0
                z[i,j,k] = Z[k] - z0
    x,y,z=np.ravel(x,order='F'),np.ravel(y,order='F'),np.ravel(z,order='F') 
    r,rt=np.sqrt(x*x+y*y+z*z),np.sqrt(x*x+y*y)
    # Normalize with geometric mean, 3/15/14 (most recently); makes error go down about order of magnitude
    rsort=np.sort(r)
    rmin=rsort[1] # first element is 0 
    rmax=rsort[len(r)-1]
    rnorm=np.sqrt(rmax*rmin)
    r=r/rnorm

    theta,phi=np.zeros(len(r)),np.zeros(len(r))
    for i in range(len(z)):
        theta[i] = mt.atan2(rt[i],z[i])
        phi[i] = mt.atan2(y[i],x[i])


    # Make the spherical harmonic matrix in sequence of [Y00,Y-11,Y01,Y11,Y-22,Y-12,Y02,Y12,Y22...]
    Yj = np.zeros((nx*ny*nz,(order+1)**2))
    fp = np.sqrt(1/(4*np.pi))
    Yj[:,0] = fp*np.sqrt(2)
    mc = 1
    for n in range(1,order+1):
        for m in range(n+1):
            ymn = r**n*lpmv(m,n,np.cos(theta))
            ymn = fp*ymn*np.sqrt((2*n+1))#/(4*np.pi))
            if m==0:
                Yj[:,mc+n] = ymn
            else: 
                # Nm is conversion factor to spherical harmonics, 
                # excluding the sqrt(2*n+1/4pi) portion so that there is no coefficient to the m=0
                N1 = float(mt.factorial(n-m))
                N2 = float(mt.factorial(n+m))
                Nm = (-1)**m*np.sqrt(2*N1/N2) 
                psin = Nm*ymn*np.sin(m*phi)
                pcos = Nm*ymn*np.cos(m*phi)
                #Yj[:,mc+1+2*(m-1)] = pcos
                #Yj[:,mc+2+2*(m-1)] = psin
                Yj[:,mc+n+m] = pcos
                Yj[:,mc+n-m] = psin
        mc += 2*n+1
    return Yj,rnorm

def spher_harm_basis_v2(r0, X, Y, Z, order):
    '''
    Computes spherical harmonics, 
    but directly by taking the real part of the spherical harmonics computed directly with scipy
   
    Returns: [Y00,Y-11,Y01,Y11,Y-22,Y-12,Y02,Y12,Y22...], rnorm
    where Yxx is a 1D array of the spherical harmonic evaluated on the grid
    rnorm is a normalization factor for the spherical harmonics
    '''

    #initialize grid with expansion point (r0) at 0
    x0,y0,z0 = r0
    y, x, z = np.meshgrid(Y-y0,X-x0,Z-z0)
    x, y, z = np.ravel(x,order='F'), np.ravel(y,order='F'), np.ravel(z,order='F')
    #change variables
    r = np.sqrt(x*x+y*y+z*z)
    r_trans = np.sqrt(x*x+y*y)
    theta = np.arctan2(r_trans,z)
    phi = np.arctan2(y,x)

    # for now normalizing as in matlab code
    dl = X[1]-X[0]
    scale = np.sqrt(np.amax(r)*dl)
    # scaling as done above
    scale2 = np.sqrt(np.amax(r)*np.amin(r))

    Yj = []

    #real part of spherical harmonics
    for l in range(0,order+1):
        for m in range(l*-1,l+1):
            Yj.append(sph_harm(m,l,theta,phi).real)

    Yj = np.transpose(Yj)

    return Yj, scale


def legendre(n,X):
    '''
    like the matlab function, returns an array of all the assosciated legendre functions of degree n
    and order m = 0,1.... n for each element in X
    '''
    r = []
    for m in range(n+1):
        r.append(lpmv(m,n,X))
    return r

def spher_harm_basis_v1a(r0, X, Y, Z, order):
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
    x, y, z = np.reshape(x,(npts)), np.reshape(y,(npts)), np.reshape(z,(npts))

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
        for m in range(2,n+2):
            c = (rs**n)*p[m-1]*np.cos((m-1)*phi)
            Q.append(c)
            cn = (rs**n)*p[m-1]*np.sin((m-1)*phi)
            Q.append(c)

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

    Yj, rnorm = spher_harm_basis_v1a(r0,X,Y,Z,order)
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

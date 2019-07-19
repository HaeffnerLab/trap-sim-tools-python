'''
Functions for performing the spherical
harmonic expansion of the potential
'''
import numpy as np
import math as mt
from scipy.special import lpmv
from scipy import linalg, matrix
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
    dl = Z[1]-Z[0]
    scale = np.sqrt(np.amax(r)*dl)
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

    Mj=np.linalg.lstsq(Yj,W) 
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

def nullspace(A,eps=1e-15):
    u,s,vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

def p2d(V,x,y):
    #fit a 2d polynomial to the potential in V
    # x and y are the 2d coordinate matrices on which the function is definied
    #Af Bf and theta are the curvatures in teh xr axis, yr axes and theta is the angle between x and xr. 

    N  = len(x)*len(y)
    con = np.ones(x.shape)

    xx,yy,xy = np.reshape(x**2,N), np.reshape(y**2,N), np.reshape(x*y,N)
    xxx,yyy,xxy,xyy = np.reshape(xx*x,N),np.reshape(yy*y,N),np.reshape(xx*y,N),np.reshape(x*yy,N)
    xxxx,yyyy,xxxy,xxyy,xyyy = np.reshape(xx**2,N),np.reshape(yy**2,N),np.reshape(xxx*y,N),np.reshape(xx*yy,N),np.reshape(x*yyy,N)
    V2 = np.reshape(V,N)
    Q = np.hstack((xxxx,yyyy,xxxy,xxyy,xyy,xxx,yyy,xxy,xy,xx,yy,xy,x,y,con))
    c = np.linalg.lstsq(Q,V2)
    c = c[0]
    theta=-0.5*np.arctan(c[11]/(c[10]-c[9]))
    Af=0.5*(c[9]*(1+1./np.cos(2*theta))+c[10]*(1-1./np.cos(2*theta)))
    Bf=0.5*(c[9]*(1-1./np.cos(2*theta))+c[10]*(1+1./np.cos(2*theta)))
    theta=180.*theta/np.pi
    return (Af, Bf, theta)






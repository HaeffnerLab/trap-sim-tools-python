import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
sys.path.append('/Users/Ben/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/ionLifetimes/bem')
from bem.formats import stl
import os
from scipy.signal import argrelextrema
from bem import Result
import pickle


def plot_RF(Result,prefix,grid):
    result = Result.from_vtk(prefix, "RF")
    p = result.potential
    maxp = np.amax(p)
    print("p max", maxp)
    x,y,z = grid.to_xyz() # p.shape[0]/2 is in the middle of x.
    p = p[p.shape[0] // 2,:,:] # get a slice of yz plane at x = p.shape[0]/2.
    print("yz plane, RF pseudo")
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 10)
    ax.set_aspect("equal")
    ax.grid(axis='both')
    yticks = np.arange(-0.1, 0.1, 0.01)
    ax.set_yticks(yticks)
    xticks = np.arange(-0.1, 0.1, 0.01)
    ax.set_xticks(xticks)
    # ax.set_ylim(0.5, 1.5)
    # ax.set_xlim(0.5,1.5)
    range = 0.35
    ax.contour(y,z, np.transpose(p), levels=np.linspace((p.max()-p.min())*range+p.min(),(p.max()-p.min())*(1-range)+p.min(), 100), cmap=plt.cm.RdYlGn)
    plt.show()

def plot_DC(Result,prefix,suffix,grid,strs,dir='x'):
    p = np.zeros(Result.from_vtk(prefix + suffix, strs[0]).potential.shape)
    for em in strs:
        ele = em
        result = Result.from_vtk(prefix + suffix, ele)
        pmid = result.potential
        maxp = np.amax(p)
        #     print("p max", maxp)
        #     print(np.shape(Vx))
        p = p+pmid
    x,y,z = grid.to_xyz()
    if dir== 'x':
        p = p[p.shape[0] // 2,:,:]
        xp = y
        yp = z
    elif dir== 'y':
        p = p[:,p.shape[1] // 2,:]
        xp = x
        yp = z
    else:
        p = p[:,:,p.shape[2] // 2]
        xp = x
        yp = y
    print("yz plane, %s potential" % ele)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    # yz plane should use x[1], x[2]. wwc
    X, Y = np.meshgrid(xp, yp)
    fig.set_size_inches(20, 10)
    ax.contour(yp,xp, p, levels=np.linspace(p.min(),p.max(), 20), cmap=plt.cm.RdYlGn)
    plt.show()

def find_saddle(V,X,Y,Z,dim, scale=1, Z0=None,min=False):
    """Returns the indices of the local extremum or saddle point of the scalar A as (Is,Js,Ks).
    V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    Z0: Z coordinate for saddle finding in a 2D potential slice
    For dim==2, the values of A are linearly extrapolated from [Z0] and [Z0]+1
    to those corresponding to Z0 and Ks is such that z[Ks]<Z0, z[Ks+1]>=Z0."""

    if (dim==2 and Z0==None):
        return 'z0 needed for evaluation'
    if dim==3:
        if len(V.shape)!=3:
            return('Problem with find_saddle.m dimensionalities.')
        # Normalize field
        f=V/float(np.amax(V))
        #next, find the gradient of our normalized potential
        #We have three output fields [Ex,Ey,Ez] from one input potential
        #because a gradient is a vector, so we associate {Ex_i,Ey_i,Ez_i} and f_i
        #for a unique location i in our 'cube' of field locations.
        #I call it 'E' because grad(V)= electric field= E
        [Ex,Ey,Ez]=np.gradient(f,abs(X[1]-X[0])/scale,abs(Y[1]-Y[0])/scale,abs(Z[1]-Z[0])/scale)

        [Ex2,Ey2,Ez2]=np.gradient(f,abs(X[1]-X[0])/scale,abs(Y[1]-Y[0])/scale,abs(Z[1]-Z[0])/scale,edge_order=2)# grid spacing is automatically consistent thanks to BEM-solver

        #return the magntiude of our gradient vector at each field point.
        E=np.sqrt(Ex**2+Ey**2+Ez**2) # magnitude of gradient (E field)

        #This is a little confusing and should be re-written (I added it after calculating the saddle)
        #Setting E=f just means we are going to calculate the global minimum potential pt.
        #Hopefully this is the the trapping location.
        if min== True:
            E=f

        #Next, we begin our search for saddle (if min=False)
        #or, we begin our search for the potential minimum (if min=True)

        #Start the search by looking at the very first location
        m=E[0,0,0]
        #meh
        origin=[0,0,0]

        #Now, we begin searching through our list of field pts (potential or |E-field|)
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                for k in range(E.shape[2]):
                    if E[i,j,k]<m:
                        # if Ex2[i,j,k]+Ey2[i,j,k]+Ez2[i,j,k]<0:
                        m=E[i,j,k]
                        origin=[i,j,k]
        #I think this checks if we need to expand our simulation....
        #but looking at looking at saddle identification, I don't think this is actually
        #working b/c the saddle is clearly out of bounds.
        if origin[0]==(0 or V.shape[0]):
            print('find_saddle: Saddle out of bounds in  x (i) direction.\n')
            return origin
        if origin[0]==(0 or V.shape[1]):
            print('find_saddle: Saddle out of bounds in  y (j) direction.\n')
            return origin
        if origin[0]==(0 or V.shape[2]):
            print('find_saddle: Saddle out of bounds in  z (k) direction.\n')
            return origin
    #################################################################################################
    if dim==2: # Extrapolate to the values of A at z0.
        V2=V
        if len(V.shape)==3:
            Ks=0 # in case there is no saddle point
            for i in range(len(Z)):
                if Z[i-1]<Z0 and Z[i]>=Z0:
                    Ks=i-1
                    if Z0<1:
                        Ks+=1
            Vs=V.shape
            if Ks>=len(Z):
                return('The selected coordinate is at the end of range.')
            v1=V[:,:,Ks]
            v2=V[:,:,Ks+1]
            V2=v1+(v2-v1)*(Z0-Z[Ks])/(Z[Ks+1]-Z[Ks])
        V2s=V2.shape
        if len(V2s)!=2: # Old: What is this supposed to check? Matlab code: (size(size(A2),2) ~= 2)
            return('Problem with find_saddle.py dimensionalities. It is {}.'.format(V2s))
        f=V2/float(np.max(abs(V2)))
        [Ex,Ey]=np.gradient(f,abs(X[1]-X[0]),abs(Y[1]-Y[0]))
        E=np.sqrt(Ex**2+Ey**2)
        m=float(np.min(E))
        mr=E[0,0]
        Is,Js=1,1 # in case there is no saddle
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                if E[i,j]<mr:
                    mr=E[i,j]
                    Is,Js=i,j
        origin=[Is,Js,Ks]
        if Is==1 or Is==V.shape[0]:
            print('find_saddle: Saddle out of bounds in  x (i) direction.\n')
            return origin
        if Js==1 or Js==V.shape[1]:
            print('find_saddle: Saddle out of bounds in  y (j) direction.\n')
            return origin
    return origin

def find_saddle_drag(V,X,Y,Z,dim, scale=1, Z0=None,min=False):
    """Returns the indices of the local extremum or saddle point of the scalar A as (Is,Js,Ks).
    V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    Z0: Z coordinate for saddle finding in a 2D potential slice
    For dim==2, the values of A are linearly extrapolated from [Z0] and [Z0]+1
    to those corresponding to Z0 and Ks is such that z[Ks]<Z0, z[Ks+1]>=Z0."""

    if (dim==2 and Z0==None):
        return 'z0 needed for evaluation'
    if dim==3:
        if len(V.shape)!=3:
            return('Problem with find_saddle.m dimensionalities.')
        # Normalize field
        E=V

        #Start the search by looking at the very first location
        m=E[0,0,0]
        #meh
        origin=[0,0,0]
        dragPath = np.zeros((len(E[:,0,0]),2))
        dragVal = np.zeros(len(E[:,0,0]))
        #Now, we begin searching through our list of field pts (potential or |E-field|)
        for i in range(E.shape[0]):
            # if Ex2[i,j,k]+Ey2[i,j,k]+Ez2[i,j,k]<0:
            minval = np.amin(E[i,:,:])
            location = np.where(E[i,:,:] == minval)
            dragPath[i] = location
            dragVal[i] = minval
        #I think this checks if we need to expand our simulation....
        #but looking at looking at saddle identification, I don't think this is actually
        #working b/c the saddle is clearly out of bounds.
        if origin[0]==(0 or V.shape[0]):
            print('find_saddle: Saddle out of bounds in  x (i) direction.\n')
            return origin
        if origin[0]==(0 or V.shape[1]):
            print('find_saddle: Saddle out of bounds in  y (j) direction.\n')
            return origin
        if origin[0]==(0 or V.shape[2]):
            print('find_saddle: Saddle out of bounds in  z (k) direction.\n')
        if min:
            idx = np.where(dragVal == np.min(dragVal))
            absmin = dragPath[idx]
            outlist = np.append(idx, absmin)
        else:
            idx = np.where(dragVal == np.max(dragVal))
            candidates = argrelextrema(dragVal, np.greater)[0]
            if len(candidates) == 0:
                if dragVal[0] < dragVal[len(dragVal)-1]:
                    idx = 0
                else:
                    idx = len(dragVal)
            if len(candidates) == 1:
                idx = candidates[0]
            else:
                for i in range(len(candidates)):
                    if i == 0:
                        idx = candidates[0]
                    elif dragVal[candidates[i]] < dragVal[idx]:
                        idx = candidates[i]
            plt.plot(dragVal)
            plt.plot(idx,dragVal[idx-1],'x')
            plt.plot(len(X)//2,V[len(X) // 2, len(Y) // 2, len(Z) // 2],'o')
            plt.show()
            sad = dragPath[idx-1]
            outlist = np.append(idx,sad)
        return outlist.astype(int)
    #################################################################################################
    if dim==2: # Extrapolate to the values of A at z0.
        V2=V
        if len(V.shape)==3:
            Ks=0 # in case there is no saddle point
            for i in range(len(Z)):
                if Z[i-1]<Z0 and Z[i]>=Z0:
                    Ks=i-1
                    if Z0<1:
                        Ks+=1
            Vs=V.shape
            if Ks>=len(Z):
                return('The selected coordinate is at the end of range.')
            v1=V[:,:,Ks]
            v2=V[:,:,Ks+1]
            V2=v1+(v2-v1)*(Z0-Z[Ks])/(Z[Ks+1]-Z[Ks])
        V2s=V2.shape
        if len(V2s)!=2: # Old: What is this supposed to check? Matlab code: (size(size(A2),2) ~= 2)
            return('Problem with find_saddle.py dimensionalities. It is {}.'.format(V2s))
        f=V2/float(np.max(abs(V2)))
        [Ex,Ey]=np.gradient(f,abs(X[1]-X[0]),abs(Y[1]-Y[0]))
        E=np.sqrt(Ex**2+Ey**2)
        m=float(np.min(E))
        mr=E[0,0]
        Is,Js=1,1 # in case there is no saddle
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                if E[i,j]<mr:
                    mr=E[i,j]
                    Is,Js=i,j
        origin=[Is,Js,Ks]
        if Is==1 or Is==V.shape[0]:
            print('find_saddle: Saddle out of bounds in  x (i) direction.\n')
            return origin
        if Js==1 or Js==V.shape[1]:
            print('find_saddle: Saddle out of bounds in  y (j) direction.\n')
            return origin
    return origin

def load_soln(file):
    with open(file) as f:
        lmid = f.read()
        lmid = lmid.split('\n')[0:168]
        lmid = np.asarray(lmid)
        lmid = lmid.astype(np.float)
        l1 = lmid.astype(np.float)
    l1[147:168] = lmid[63:84]
    l1[105:126] = lmid[84:105]
    l1[63:84] = lmid[105:126]
    l1[84:105] = lmid[126:147]
    l1[126:147] = lmid[147:168]
    return l1
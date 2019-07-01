"""
simulation.py

Container class for post-processing
resuls from BEMSolver simulations.

"""

import numpy as np
import expansion as e
import pickle
import optimsaddle as o

class simulation:

    def __init__(self):
        self.electrode_grad = []
        self.electrode_hessian = []
        self.electrode_multipole = []
        self.RF_null = []

    def import_data(self, path, numElectrodes, na, perm, saveFile = None):
        '''
        path: file path to the potential data
        pointsPerAxis: number of data points in each axis of the simulation grid
        numElectrodes: number of electrodes in the simulation
        saveFile: optional file to save the output dictionary
        
        adds to self X, Y, Z axes, name, position, and potentials for each electrode, max & min electrode position for used electrodes
        '''
        try:
            f = open(path,'rb')
        except IOError:
            return ('No pickle file found.')
        trap = pickle.load(f)

        Xi = trap['X'] #sandia defined coordinates
        Yi = trap['Y']
        Zi = trap['Z']
        #get everything into expected coordinates (described in project_paramters)
        coords = [Xi,Yi,Zi]
        X = coords[perm[0]]
        Y = coords[perm[1]]
        Z = coords[perm[2]]

        self.X, self.Y, self.Z = X,Y,Z
        self.nx, self.ny, self.nz = len(X), len(Y), len(Z)
        
        self.numElectrodes = numElectrodes

        #truncate axial direction to only care about part that is spanned by electrodes
        pos = []
        for key in trap['electrodes']:
            p = trap['electrodes'][key]['position']
            pos.append(p)
        xs = [p[0] for p in pos]
        self.Z_max = max(xs)/1000.0
        self.Z_min = min(xs)/1000.0

        I_max = np.abs(Z-self.Z_max).argmin() + 20
        print I_max 

        I_min = np.abs(Z-self.Z_min).argmin() - 20
        self.Z = self.Z[I_min:I_max]
        self.nz = I_max-I_min
        self.npts = self.nx*self.ny*self.nz

        self.electrode_potentials = []
        self.electrode_names = []
        self.electrode_positions = []

        for key in trap['electrodes']:
            Vs = trap['electrodes'][key]['V']
            Vs = Vs.reshape(na[0],na[1],na[2])
            Vs = np.transpose(Vs,perm)
            Vs = Vs[:,:,I_min:I_max]

            self.electrode_names.append(trap['electrodes'][key]['name'])
            if trap['electrodes'][key]['name'] == 'RF':
                self.RF_potential = Vs
            self.electrode_positions.append(trap['electrodes'][key]['position'])
            self.electrode_potentials.append(Vs)

        return

    def compute_gradient(self):
        # computes gradient & hessian of potential
        if len(self.electrode_grad) != 0:
            print "gradient already computed"
            return
        else:
            nx = len(self.X)
            ny = len(self.Y)
            nz = len(self.Z)

            self.electrode_grad = np.empty((self.numElectrodes, 3, nx,ny,nz))
            self.electrode_hessian = np.empty((self.numElectrodes,3,3,nx,ny,nz))

            for i, el in enumerate(self.electrode_potentials):
                grad,hessian = e.compute_gradient(el,nx,ny,nz)
                self.electrode_grad[i,:,:,:,:] = grad
                self.electrode_hessian[i,:,:,:,:,:] = hessian
        return

    def compute_multipoles(self):
        if len(self.electrode_multipole) != 0:
            print "multipoles already computed"
            return
        else:
            for i,el in enumerate(self.electrode_potentials):
                g = self.electrode_grad[i]
                h = self.electrode_hessian[i]
                self.electrode_multipole.append(e.compute_multipoles(g,h))
        return

    def expand_potentials_spherHarm(self):
        '''
        Computes a multipole expansion of every electrode around the specified value expansion_point to the specified order.
        Defines the class variables:
        (1) self.multipole_expansions [:, el] = multipole expansion vector for electrode el
        (2) self.regenerated potentials [:,el] = potentials regenerated from s.h. expansion
        
        '''

        N = (self.expansion_order + 1)**2 # number of multipoles

        self.multipole_expansions = np.zeros((N, self.numElectrodes))
        self.electrode_potentials_regenerated = np.zeros(np.array(self.electrode_potentials).shape)

        X, Y, Z = self.X, self.Y, self.Z

        for el in range(self.numElectrodes):

            #multipole expansion
            potential_grid = self.electrode_potentials[el]
            Mj,Yj,scale = e.spher_harm_expansion(potential_grid, self.expansion_point, X, Y, Z, self.expansion_order)
            self.multipole_expansions[:, el] = Mj[0:N].T

            #regenerated field
            Vregen = e.spher_harm_cmp(Mj,Yj,scale,self.expansion_order)
            self.electrode_potentials_regenerated[el] = Vregen.reshape([self.nx,self.ny,self.nz])

            if self.electrode_names[el] == 'RF':
                self.RF_multipole_expansion = Mj[0:N].T
                self.RF_potential_regenerated = Vregen.reshape([self.nx,self.ny,self.nz])

        return

    def rf_saddle (self,expansion_point,order):
        ## finds the rf_saddle point near the desired expansion point and updates the expansion_position

        N = (order + 1)**2 # number of multipoles

        Mj,Yj,scale = e.spher_harm_expansion(self.RF_potential, expansion_point, self.X, self.Y, self.Z, order)
        self.RF_multipole_expansion = Mj[0:N].T

        #regenerated field
        Vregen = e.spher_harm_cmp(Mj,Yj,scale,order)
        self.RF_potential_regenerated = Vregen.reshape([self.nx,self.ny,self.nz])

        [Xrf,Yrf,Zrf] = o.exact_saddle(self.RF_potential,self.X,self.Y,self.Z,2,Z0=expansion_point[2])
        [Irf,Jrf,Krf] = o.find_saddle(self.RF_potential,self.X,self.Y,self.Z,2,Z0=expansion_point[2])
        print [Xrf,Yrf,Zrf]
        print [Irf,Jrf,Krf]

        self.expansion_point = [Xrf,Yrf,Zrf]
        print self.expansion_point
        self.expansion_order = order

        return

    def set_controlled_electrodes(self, controlled_electrodes, shorted_electrodes = []):
        '''
        Define the set of electrodes under DC control

        controlled_electrodes: list of integers specifying the electrodes to be controlled, in the appropriate
        order for the control matrix

        shorted_electrodes: optional. list of electrodes shorted together. Form: [(a, b), (c, d, e), ...]

        If some electrodes are shorted, only use one of each set in controlled_electrodes.
        '''

        M_shorted = self.multipole_expansions.copy()
        N = M_shorted[:,0].shape[0] # length of the multipole expansion vector
        for s in shorted_electrodes:
            vec = np.zeros(N)
            for el in s:
                vec += self.multipole_expansions[:, el]
            [M_shorted[:, el]] = [vec for el in s]

        # multipole expansion matrix after accounting for shorted electrodes
        # and uncontrolled electrodes
        self.reduced_multipole_expansions = np.zeros((N, len(controlled_electrodes)))
        for k, el in enumerate(controlled_electrodes):
            self.reduced_multipole_expansions[:, k] = M_shorted[:,el]






### TESTING    
from matplotlib import pyplot as plt
path = '../HOA_trap/CENTRALonly.pkl'
na = [941,13,15]
ne = 13
perm = [1,2,0]
position = [0,0.07,0]

s = simulation()
s.import_data(path,ne,na,perm)
s.rf_saddle(position,2)
#s.compute_gradient()
#s.compute_multipoles()

s.expand_potentials_spherHarm()
Nmulti = len(s.multipole_expansions)
Nelec = s.numElectrodes

#plotting potentials
fig,ax = plt.subplots(2,1)
for n in range(Nelec):
   if s.electrode_names[n] != "RF":
       ax[0].plot(s.Z,s.electrode_potentials[n][6][7],label = str(s.electrode_names[n]))
       ax[1].plot(s.Z,s.electrode_potentials_regenerated[n][6][7],label = str(s.electrode_names[n]))
ax[0].legend()
ax[1].legend()
plt.show()


# #plotting gradients
# fig,ax = plt.subplots(3,1)
# for n in range(Nelec):

#    if s.electrode_names[n] != "RF":
#         for i in range(3):
#             ax[i].plot(s.Z,s.electrode_grad[n][i][6][7],label = str(s.electrode_names[n])+'grad'+str(i))
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# plt.show()

#plotting multipoles
#fig,ax = plt.subplots(9,3)
#for n in range(Nelec):
#    if s.electrode_names[n] != "RF":





###################################################################
#checking spherical harmonic basis
Yj1, rnorm1 = e.spher_harm_basis(position, s.X, s.Y, s.Z, 2)

y, x, z = np.meshgrid(s.Y,s.X,s.Z)

x, y, z = np.reshape(x,(s.npts)), np.reshape(y,(s.npts)), np.reshape(z,(s.npts))

fig,ax = plt.subplots(9,3)

for n in range(9):
    Y3 = Yj1[:,n].reshape(s.nx,s.ny,s.nz)
    a = ax[n][0].imshow(Y3[:,5,:])
    b = ax[n][1].imshow(Y3[6,:,:])
    c = ax[n][2].imshow(Y3[:,:,48])
    fig.colorbar(a,ax=ax[n][0])
    fig.colorbar(b,ax=ax[n][1])
    fig.colorbar(b,ax=ax[n][2])

plt.show()
    

############################################################
# plotting multipole coefficients 
fig,ax = plt.subplots(6,1)
for n in range(Nelec):
    if (s.electrode_names[n] in ['Q19','Q20']):
        ax[0].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))
    if (s.electrode_names[n] in ['Q21','Q22']):
        ax[1].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))
    if (s.electrode_names[n] in ['Q23','Q24']):
        ax[2].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))
    if (s.electrode_names[n] in ['Q17','Q18']):
        ax[3].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))
    if (s.electrode_names[n] in ['Q15','Q16']):
        ax[4].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))
    if (s.electrode_names[n] in ['Q39','Q40']):
        ax[5].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[4].legend()
ax[5].legend()

plt.show()
"""
simulation.py

Container class for post-processing
resuls from BEMSolver simulations.

"""

import numpy as np
import expansion as e
import pickle

class simulation:

    def __init__(self, scale):
        self.scale = scale
        self.electrode_grad = []
        self.electrode_hessian = []
        self.electrode_multipole = []

    def import_data(self, path, numElectrodes, na, perm, saveFile = None):
        '''
        path: file path to the potential data
        pointsPerAxis: number of data points in each axis of the simulation grid
        numElectrodes: number of electrodes in the simulation
        saveFile: optional file to save the output dictionary
        
        adds to self X, Y, Z axes as well as the potential grids for each electrode
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
        X = coords[perm[0]]/self.scale
        Y = coords[perm[1]]/self.scale
        Z = coords[perm[2]]/self.scale

        self.X, self.Y, self.Z = X,Y,Z
        self.numElectrodes = numElectrodes

        self.electrode_potentials = []
        self.electrode_names = []
        self.electrode_positions = []

        for key in trap['electrodes']:
            Vs = trap['electrodes'][key]['V']
            Vs = Vs.reshape(na[0],na[1],na[2])
            Vs = np.transpose(Vs,perm)

            self.electrode_names.append(trap['electrodes'][key]['name'])
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


    def expand_potentials_spherHarm(self, expansion_point, order=3):
        '''
        Computes a multipole expansion of every electrode
        around the specified value expansion_point.

        expansion_point: list [x0, y0, z0] of location
        to compute the fiel expansion
        order: int, order of the expansion to carry out

        returns: matrix of multipole expansions for each electrodes
        matrix[:, el] = multipole expansion vector for electrode el
        
        Defines the class variable self.multipole_expansions
        '''

        N = (order + 1)**2 # number of multipoles

        self.multipole_expansions = np.zeros((N, self.numElectrodes))
        self.expansion_order = order

        X, Y, Z = self.X, self.Y, self.Z

        for el in range(self.numElectrodes):
            potential_grid = self.electrode_potentials[el]
            vec = e.spher_harm_expansion(potential_grid, expansion_point, X, Y, Z, order)
            self.multipole_expansions[:, el] = vec[0:N].T

        return self.multipole_expansions

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

    def generate_control_matrix(self, controlled_multipoles):
        '''
        Generates the multipole control matrix

        controlled_multipoles: list of integers specifying which multipoles are to be controlled
        0, 1, 2 correspond to Ex, Ey, Ez, 3-8 correspond to to the quadrupoles
        
        '''

        multipoles = np.zeros((self.expansion_order+1)**2)
        for k in controlled_multipoles:
            multipoles[k] = 1




### TESTING    
from matplotlib import pyplot as plt
path = '../HOA_trap_v1/CENTRALonly.pkl'
na = [941,13,15]
ne = 13
perm = [1,2,0]
position = [0,0,0]

s = simulation(1)
s.import_data(path,ne,na,perm)
s.compute_gradient()
s.compute_multipoles()

#s.expand_potentials(position,2)
#Nmulti = len(s.multipole_expansions)
Nelec = s.numElectrodes

#plotting potentials
fig,ax = plt.subplots(1,1)
for n in range(Nelec):
   if s.electrode_names[n] != "RF":
       ax.plot(s.Z,s.electrode_potentials[n][6][7],label = str(s.electrode_names[n]))
ax.legend()
plt.show()


#plotting gradients
fig,ax = plt.subplots(3,1)
for n in range(Nelec):

   if s.electrode_names[n] != "RF":
        for i in range(3):
            ax[i].plot(s.Z,s.electrode_grad[n][i][6][7],label = str(s.electrode_names[n])+'grad'+str(i))
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()

#plotting multipoles
fig,ax = plt.subplots(9,3)
for n in range(Nelec):
    if s.electrode_names[n] != "RF":





# ###################################################################
# #checking spherical harmonic basis
# Yj1, rnorm1 = e.spher_harm_basis(position, s.X, s.Y, s.Z, 2)

# y, x, z = np.meshgrid(s.Y,s.X,s.Z)
# npts = 941*13*15
# x, y, z = np.reshape(x,(npts)), np.reshape(y,(npts)), np.reshape(z,(npts))

# fig,ax = plt.subplots(9,3)

# for n in range(9):
#     Y3 = Yj1[:,n].reshape(na[1],na[2],na[0])
#     a = ax[n][0].imshow(Y3[:,6,460:480])
#     b = ax[n][1].imshow(Y3[7,:,460:480])
#     c = ax[n][2].imshow(Y3[:,:,470])
#     fig.colorbar(a,ax=ax[n][0])
#     fig.colorbar(b,ax=ax[n][1])
#     fig.colorbar(b,ax=ax[n][2])

# plt.show()

# ############################################################
# # plotting multipole coefficients 
# fig,ax = plt.subplots(6,1)
# for n in range(Nelec):
#     if (s.electrode_names[n] in ['Q19','Q20']):
#         ax[0].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))
#     if (s.electrode_names[n] in ['Q21','Q22']):
#         ax[1].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))
#     if (s.electrode_names[n] in ['Q23','Q24']):
#         ax[2].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))
#     if (s.electrode_names[n] in ['Q17','Q18']):
#         ax[3].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))
#     if (s.electrode_names[n] in ['Q15','Q16']):
#         ax[4].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))
#     if (s.electrode_names[n] in ['Q39','Q40']):
#         ax[5].plot(range(Nmulti),s.multipole_expansions[:,n],'x',label = str(s.electrode_names[n]))

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[3].legend()
# ax[4].legend()
# ax[5].legend()

# plt.show()
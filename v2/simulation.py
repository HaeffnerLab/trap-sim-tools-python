"""
simulation.py

Container class for post-processing
resuls from BEMSolver simulations.

"""

import numpy as np
import expansion as e
import pickle
import optimsaddle as o
import matplotlib.pyplot as plt
from matplotlib import cm

class simulation:

    def __init__(self,charge, mass, useRF = False):
        self.electrode_grad = []
        self.electrode_hessian = []
        self.electrode_multipole = []
        self.RF_null = []
        self.multipole_expansions = []
        self.instances = []

        self.charge = charge
        self.mass = mass
        self.useRF = useRF

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
            print 'No pickle file found.'
            return
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

            
            if trap['electrodes'][key]['name'] == 'RF':
                self.RF_potential = Vs
                if self.useRF:
                    self.electrode_names.append(trap['electrodes'][key]['name'])
                    self.electrode_positions.append(trap['electrodes'][key]['position'])
                    self.electrode_potentials.append(Vs)
            else:
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

    def expand_potentials_spherHarm(self):
        '''
        Computes a multipole expansion of every electrode around the specified value expansion_point to the specified order.
        Defines the class variables:
        (1) self.multipole_expansions [:, el] = multipole expansion vector for electrode el
        (2) self.regenerated potentials [:,el] = potentials regenerated from s.h. expansion
        
        '''

        N = (self.expansion_order + 1)**2 # number of multipoles for expansion (might be less than for regeneration)
        order = max(self.expansion_order,self.regeneration_order)

        self.multipole_expansions = np.zeros((N, self.numElectrodes))
        self.electrode_potentials_regenerated = np.zeros(np.array(self.electrode_potentials).shape)

        X, Y, Z = self.X, self.Y, self.Z

        for el in range(self.numElectrodes):

            #multipole expansion
            potential_grid = self.electrode_potentials[el]
            Mj,Yj,scale = e.spher_harm_expansion(potential_grid, self.expansion_point, X, Y, Z, order)
            self.multipole_expansions[:, el] = Mj[0:N].T

            #regenerated field
            Vregen = e.spher_harm_cmp(Mj,Yj,scale,self.regeneration_order)
            self.electrode_potentials_regenerated[el] = Vregen.reshape([self.nx,self.ny,self.nz])

            if self.electrode_names[el] == 'RF':
                self.RF_multipole_expansion = Mj[0:N].T
                #self.RF_potential_regenerated = Vregen.reshape([self.nx,self.ny,self.nz])

        return

    def rf_saddle (self):
        ## finds the rf_saddle point near the desired expansion point and updates the expansion_position

        N = (self.expansion_order + 1)**2 # number of multipoles for expansion (might be less than for regeneration)
        order = max(self.expansion_order,self.regeneration_order)

        Mj,Yj,scale = e.spher_harm_expansion(self.RF_potential, self.expansion_point, self.X, self.Y, self.Z, order)
        self.RF_multipole_expansion = Mj[0:N].T

        #regenerated field
        Vregen = e.spher_harm_cmp(Mj,Yj,scale,self.regeneration_order)
        self.RF_potential_regenerated = Vregen.reshape([self.nx,self.ny,self.nz])

        [Xrf,Yrf,Zrf] = o.exact_saddle(self.RF_potential,self.X,self.Y,self.Z,2,Z0=self.expansion_point[2])
        [Irf,Jrf,Krf] = o.find_saddle(self.RF_potential,self.X,self.Y,self.Z,2,Z0=self.expansion_point[2])

        self.expansion_point = [Xrf,Yrf,Zrf]

        return

    def expand_field(self,expansion_point,expansion_order,regeneration_order):

        self.expansion_point = expansion_point
        self.expansion_order = expansion_order
        self.regeneration_order = regeneration_order
        self.rf_saddle()
        self.expand_potentials_spherHarm()

        return


    def set_controlled_electrodes(self, controlled_electrodes, shorted_electrodes = []):
        '''
        XXXXXXXXXXXXXXXXXXXXXXXxNEED TO REWRITEXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx
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
    

    def set_used_multipoles(self,multipoles_toUse):
        ## keep only the multipoles used
        ## used_multipoles is a list of 0 or 1 the length of the # of multipoles
        if len(self.multipole_expansions) == 0:
            print "must expand the potential first"
            return

        used_multipoles = []
        for i,b in enumerate(multipoles_toUse):
            if b:
                used_multipoles.append(self.multipole_expansions[i,:])
        self.multipole_expansions = used_multipoles

        return

    def multipole_control(self,regularize):
        ## inverts the multipole coefficient array to get the multipole controls
        ## (e.g. voltages)
        if len(self.multipole_expansions) == 0:
            print "must expand the potential first"
            return
        M = len(self.multipole_expansions)
        E = len(self.electrode_potentials)
        self.multipoleControl = []
        for i in range(M):
            B = np.zeros(M)
            B[i] = 1
            A = np.linalg.lstsq(self.multipole_expansions,B)
            self.multipoleControl.append(A[0])
        #check nullspace & regularize if necessary
        if M < E:
            K = e.nullspace(self.multipole_expansions)
        else:
            print 'There is no nullspace because the coefficient matrix is rank deficient. \n There can be no regularization.'
            K = None
            regularize = False
        if regularize:
            for i in range(M):
                Cv = self.multipoleControl[:,i].T
                L = np.linalg.lstsq(K,Cv)
                test = np.dot(K,L[0])
                self.multipoleControl[i] = self.multipoleControl[i]-test
        
        return

    def print_cFile(self,fName=None):
        #prints the current c file to a file for labrad to use. 
        #takes an optional file name, otherwise just saves it as Cfile.txt
        #just a text file with a single column with (# rows) = (# electrodes)*(# multipoles)
        if fName == None:
            fName = 'Cfile.txt'
        f = open(fName,'w')
        for i in range(len(self.multipoleControl)):
            np.savetxt(f, self.multipoleControl[i], delimiter=",")
        print self.electrode_names
        f.close()

        return



    def setVoltages(self,coeffs,name = None):
        # takes a set of desired multipole coefficients and returns the voltages needed to acheive that.
        # creates an instance that will be used to save trap attributes for different configurations
        voltages = np.dot(np.array(self.multipoleControl).T,coeffs)
        instance = {'name':name,'coeffs':coeffs,'voltages':voltages,'potential':False,'U':False}
        self.instances.append(instance)

        return voltages

    ###########################TO BE IMPLEMENTED #####################################
    # def dcPotential(self):
    #     # calculates the dc potential given the applied voltages for all instances that don't have a dcpotential yet. 
    #     #### should add stray field.
    #     for instance in self.instances:
    #         if not instance['potential']:
    #             potential = np.zeros((self.nx,self.ny,self.nz))
    #             for i in range(self.numElectrodes):
    #                 potential = potential + instance['voltages'][i]*self.electrode_potentials[i]
    #             instance['potential'] = [potential]
    #     return

    # def post_process(self):
    #     # finds the secular frequencies, tilt angle, and position of the dc saddle point
    #     dx = self.X[1]-self.X[0]
    #     dy = self.Y[1]-self.Y[0]
    #     dz = self.Z[1]-self.Z[0]
    #     y, x, z = np.meshgrid(self.Y,self.X,self.Z)

    #     [Ex_RF,Ey_RF,Ez_RF] = np.gradient(self.RF_potential,dx,dy,dz)
    #     Esq_RF = Ex**2 + Ey**2 + Ez**2
    #     [Irf,Jrf,Krf] = o.find_saddle(self.RF_potential,self.X,self.Y,self.Z,2,Z0=self.expansion_point[2])

    #     PseudoPhi = Esq*(self.charge**2)/(4*self.mass*self.Omega**2)
    #     for instance in self.instances:
    #         if not instance['U']:
    #             [fx,fy,fz] = e.trapFrequencies
    #             U = PseudoPhi + self.charge*instance['potential']
    #             instance['U'] = U
    #             Uxy = U[Irf-5:Irf+5,Jrf-5:Jrf+5,Krf]
    #             maxU = np.amax(Uxy)
    #             Uxy = Uxy/MU
    #             dl = dx*5 ## not sure why this scaling exists.
    #             xr = (self.X[Irf-5:Irf+5,Jrf-5:Jrf+5,Krf]-self.X[Irf,Jrf,Krf])/dl
    #             yr = (self.Y[Irf-5:Irf+5,Jrf-5:Jrf+5,Krf]-self.Y[Irf,Jrf,Krf])/dl
    #             C1,C2,theta = p2d(Uxy,xr,yr)
    #             instance['fx'] = 1e3*np.sqrt
    #             instance


    def plot_multipoleCoeffs(self):
        #plots the multipole coefficients for each electrode - should be changed to be less specific
        fig,ax = plt.subplots(1,1)
        Nelec = self.numElectrodes
        Nmulti = len(self.multipole_expansions)
        for n in range(Nelec):
                ax.plot(range(Nmulti),self.multipole_expansions[:,n],'x',label = str(self.electrode_names[n]))
        ax.legend()
        plt.show()
        return

    def plot_trapV(self,V,title=None):
        #plots trap voltages (V) (e.g. for each multipole, or for final trapping configuration)
        fig,ax  = plt.subplots(1,1)
        xpos = [p[0] for p in self.electrode_positions]
        ypos = [p[1] for p in self.electrode_positions]
        plot = ax.scatter(xpos, ypos, 500, V)
        fig.colorbar(plot)
        plt.title(title)
        plt.show()
        return

    # def plot_potential(self,potential, title=None)
    #     # plots a give potential ... need to implement
    #     return





### TESTING    

path = '../HOA_trap_v1/CENTRALonly.pkl'
na = [941,13,15] # number of points per axis
ne = 12 # number of electrodes
perm = [1,2,0] #in this code y(coord 2)- height, z(coord 3) - axial, x(coord 1) - radial
position = [0,0.07,0] 

charge = 1.6021764e-19
mass = 1.67262158e-27 * 40
RF_ampl = 100 
RF_freq = 50e6 #in Hz


s = simulation(charge,mass)
s.import_data(path,ne,na,perm)
s.expand_field(position,2,2)

#plotting potentials
fig,ax = plt.subplots(2,1)
for n in range(len(s.electrode_positions)):
    ax[0].plot(s.Z,s.electrode_potentials[n][6][7],label = str(s.electrode_names[n]))
    ax[1].plot(s.Z,s.electrode_potentials_regenerated[n][6][7],label = str(s.electrode_names[n]))
ax[0].legend()
ax[1].legend()
plt.show()

s.plot_multipoleCoeffs()

usedMultipoles = np.zeros((s.expansion_order+1)**2)
usedMultipoles[0:6] = np.ones(6)
s.set_used_multipoles(usedMultipoles)
s.multipole_control(False)

print s.multipoleControl


for n in range(len(s.multipoleControl)):
    s.plot_trapV(s.multipoleControl[n],"Multipole " + str(n))

s.print_cFile('test2.cls') # file includes *all* multipoles that should be available for the user in the gui.

vs_solution1 = np.zeros(ne)
vs_solution1[1] = -0.91
vs_solution1[3] = -1
vs_solution1[10] = -1
vs_solution1[9] = -0.919
coeffs_solution1 = np.dot(s.multipole_expansions,vs_solution1)
print coeffs_solution1









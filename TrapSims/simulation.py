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
from scipy.optimize import curve_fit

class simulation:

    def __init__(self,charge, mass, useRF = False):
        self.electrode_grad = []
        self.electrode_hessian = []
        self.electrode_multipole = []
        self.RF_null = []
        self.multipole_expansions = []

        self.charge = charge
        self.mass = mass
        self.useRF = useRF

    def import_data(self, path, numElectrodes, na, perm):
        '''
        path: file path to the potential data. This should be a .pkl file containing a dictionary with the following keys:
            trap['X'] = X values 
            trap['Y'] = Y values
            trap['Z'] = Z values
            trap['electrodes'] = dictionary with one entry for each electrode. each electrode has the following keys. 
                We're using Q14 as an example of an electrode key
                trap['electrodes']['Q14'][name] = 'Q14'
                trap['electrodes']['Q14'][position] = position on the chip (just used for pretty plotting)
                trap['electrodes']['Q14']['V'] = grid of potential values (should be a nx*ny*nz by 1 vector that will be reshaped properly)
        
        For HOA traps, this .pkl file can be made with the jupyter notebook import_data_HOA

        For other traps, other import functions could be added.

        na: number of data points in each axis of the simulation grid
        numElectrodes: number of electrodes in the simulation
        perm: permutation to get axes in the potential file to match axes assumed in this code:
            in this code, coordinates are [radial, height, axial]
            for the HOA, coordinates are [axial, radial, height] so perm would be [1,2,0]
        
        adds to self X, Y, Z axes, name, position, and potentials for each electrode, max & min electrode position for used electrodes
        '''
        try:
            f = open(path,'rb')
        except IOError:
            print 'ERROR: No pickle file found.'
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


    def expand_potentials_spherHarm(self,ROI=None):
        '''
        ROI = region to expand the potential around. 
        Computes a multipole expansion of every electrode around the specified value expansion_point to the specified order.
        Defines the class variables:
        (1) self.multipole_expansions [:, el] = multipole expansion vector for electrode el
        (2) self.regenerated potentials [:,el] = potentials regenerated from s.h. expansion

        The function returns the coefficients in the order:[C00 C10 C11c C11s ]'
        These correspond to the multipoles in cartesian coordinares: 
        [c z -x -y (z^2-x^2/2-y^2/2) -6zx -6yz 6x^2-6y^2 12xy ...
         1 2  3  4       5             6    7     8       9  ...

        Or in terms of the Littich thesis:
        M1 M3 M4 M2 M7 M8 M6 M9 M5 (Using the convention in G. Littich's master thesis (2011))
        0  1  2  3  4  5  6  7  8  (the ith component of Q matrix)
        
        '''

        N = (self.expansion_order + 1)**2 # number of multipoles for expansion (might be less than for regeneration)
        order = self.expansion_order

        if ROI == None:
        	self.nx_trunc = self.nx
        	self.ny_trunc = self.ny
        	self.nz_trunc = self.nz
        	self.X_trunc = self.X
        	self.Y_trunc = self.Y
        	self.Z_trunc = self.Z
        else:
    	    self.X_trunc = self.X[self.expansion_coords[0]-ROI[0]:self.expansion_coords[0]+ROI[0]]
    	    self.Y_trunc = self.Y[self.expansion_coords[1]-ROI[1]:self.expansion_coords[1]+ROI[1]]
    	    self.Z_trunc = self.Z[self.expansion_coords[2]-ROI[2]:self.expansion_coords[2]+ROI[2]]
    	    self.nx_trunc = ROI[0]*2
    	    self.ny_trunc = ROI[1]*2
    	    self.nz_trunc = ROI[2]*2

        self.multipole_expansions = np.zeros((N, self.numElectrodes))
        self.electrode_potentials_regenerated = np.zeros([self.numElectrodes,self.nx_trunc,self.ny_trunc,self.nz_trunc])

        for el in range(self.numElectrodes):

            #multipole expansion
            if ROI == None:
            	potential_grid = self.electrode_potentials[el]

            else:
            	potential_grid = self.electrode_potentials[el][self.expansion_coords[0]-ROI[0]:self.expansion_coords[0]+ROI[0],
            												self.expansion_coords[1]-ROI[1]:self.expansion_coords[1]+ROI[1],
            												self.expansion_coords[2]-ROI[2]:self.expansion_coords[2]+ROI[2]]


            Mj,Yj,scale = e.spher_harm_expansion(potential_grid, self.expansion_point, self.X_trunc, self.Y_trunc, self.Z_trunc, order)
            self.multipole_expansions[:, el] = Mj[0:N].T

            #regenerated field
            
            Vregen = e.spher_harm_cmp(Mj,Yj,scale,order)
            self.electrode_potentials_regenerated[el] = Vregen.reshape([self.nx_trunc,self.ny_trunc,self.nz_trunc])


            if self.electrode_names[el] == 'RF':
                self.RF_multipole_expansion = Mj[0:N].T
        self.multipoles = Yj
        return

    def rf_saddle (self):
        ## finds the rf_saddle point near the desired expansion point and updates the expansion_position

        N = (self.expansion_order + 1)**2 # number of multipoles for expansion 
        order = self.expansion_order

        Mj,Yj,scale = e.spher_harm_expansion(self.RF_potential, self.expansion_point, self.X, self.Y, self.Z, order)
        self.RF_multipole_expansion = Mj[0:N].T

        #regenerated field
        Vregen = e.spher_harm_cmp(Mj,Yj,scale,order)
        self.RF_potential_regenerated = Vregen.reshape([self.nx,self.ny,self.nz])

        [Xrf,Yrf,Zrf] = o.exact_saddle(self.RF_potential,self.X,self.Y,self.Z,2,Z0=self.expansion_point[2])
        [Irf,Jrf,Krf] = o.find_saddle(self.RF_potential,self.X,self.Y,self.Z,2,Z0=self.expansion_point[2])

        self.expansion_point = [Xrf,Yrf,Zrf]
        self.expansion_coords = [Irf,Jrf,Krf]

        return

    def expand_field(self,expansion_point,expansion_order,ROI=None):

        self.expansion_point = expansion_point
        self.expansion_order = expansion_order
        self.rf_saddle()
        self.expand_potentials_spherHarm(ROI)

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
            print "ERROR: must expand the potential first"
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
            print "ERROR: must expand the potential first"
            return
        M = len(self.multipole_expansions)
        E = len(self.electrode_potentials)
        self.multipoleControl = []
        for i in range(M):
            B = np.zeros(M)
            B[i] = 1
            A = np.linalg.lstsq(self.multipole_expansions,B,rcond=None)
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
                Cv = self.multipoleControl[i].T
                L = np.linalg.lstsq(K,Cv,rcond=None)
                test = np.dot(K,L[0])
                self.multipoleControl[i] = self.multipoleControl[i]-test
        
        return

    def print_cFile(self, fName=None):
        '''
        prints the current c file to a file for labrad to use, 
            just a text file with a single column with (# rows) = (# electrodes)*(# multipoles) 

        takes an optional file name, otherwise just saves it as Cfile.txt

        PRINTS THEM OUT IN ALPHABETICAL ORDER (THIS IS HOW NEW DAC CONTROL TAKES THE NAMES)

        '''
        if fName == None:
            fName = 'Cfile.txt'
        f = open(fName,'w')
        print self.electrode_names
        indices = np.argsort(self.electrode_names)
        print indices
        mC = np.array(self.multipoleControl)
        for j in range(len(self.multipoleControl)):
            for i in indices:
                print self.electrode_names[i]
                np.savetxt(f, [mC[j,i]], delimiter=",")
        f.close()

        return

    def setVoltages(self,coeffs,name = None):
        # takes a set of desired multipole coefficients and returns the voltages needed to acheive that.
        # creates an instance that will be used to save trap attributes for different configurations
        voltages = np.dot(np.array(self.multipoleControl).T,coeffs)

        return voltages

    def dcPotential(self,vs):
        # calculates the dc potential given the applied voltages
        potential = np.zeros((self.nx,self.ny,self.nz))
        for i in range(self.numElectrodes):
                potential = potential + self.electrode_potentials[i]*vs[i]
        self.dc_potential = potential
        return

    def regenPotential(self,vs):
    	#calculates the approximate dc potential using the regenerated potentials from the harmonic expansion
    	potential = np.zeros((self.nx_trunc,self.ny_trunc,self.nz_trunc))
        for i in range(self.numElectrodes):
        	potential = potential + self.electrode_potentials_regenerated[i]*vs[i]
        self.regen_potential = potential
        return

    def post_process(self,vs,Omega,RF_amplitude, trunc = False, ROI=None):
        '''
        inputs: 
        dcVoltages = voltages to apply to each dc electrode (and RF if used in the simulation)
        Omega = RF drive frequency
        ROI = number of points to consider to left and right of the center point

        finds the axial frequency of the actual potential, the regenerated potential, and from the U2 component directly.
        '''

        self.dcPotential(vs)
        self.regenPotential(vs)

        [Irf,Jrf,Krf] = self.expansion_coords

        #find axial trap frequency from exact potential & regnerated potential
        if trunc == True:
        	z = self.Z_trunc
        else:
        	z = self.Z
        if ROI == None:
            Uz_dc = self.dc_potential[Irf,Jrf,:]
            Uz_regen = self.regen_potential[Irf,Jrf,:]
        else:
            Uz_dc = self.dc_potential[Irf,Jrf,Krf-ROI:Krf+ROI]
            Uz_regen = self.regen_potential[Irf,Jrf,Krf-ROI:Krf+ROI]
            z = z[Krf-ROI:Krf+ROI]

        c_dc = np.polyfit(z,Uz_dc,2)
        self.fz_dc = 1e-6*np.sqrt(2*c_dc[0]*1e6*self.charge/self.mass)/(2*np.pi) # in MHz
        print 'actual axial trap frequency: ',self.fz_dc, 'MHz'


        #c_regen = np.polyfit(self.Z_trunc,Uz_regen,2)
        #self.fz_regen = 1e-6*np.sqrt(2*c_regen[0]*1e6*self.charge/self.mass)/(2*np.pi) # in MHz
        #print 'axial trap frequency of regenerated potential: ',self.fz_regen, 'MHz'

        #find axial trap frequency via U2 coefficient 
        ## write now this is hard coded because I know U2 is the 3rd coefficient
        coeffs = np.dot(self.multipole_expansions,vs)
        self.fz_multipole = 1e-6*np.sqrt(2*coeffs[3]*1e6*self.charge/self.mass)/(2*np.pi) # in MHz
        print 'axial trap frequency as determined by U2 coefficient: ', self.fz_multipole, 'MHz'


       	return self.fz_dc

    ### plotting helper functions
    def plot_multipoleCoeffs(self,vs,names = None):
        #plots the multipole coefficients for each electrode
        multipole_names = ['C','Ez (axial)','Ex (radial)', 'Ey (height)',r'U2=z^2-(x^2+y^2)/2',
        					'U5=-3zx', 'U4=-3yz', r'U1=3(x^2-y^2)', 'U3=6xy']

        Nmulti = len(self.multipole_expansions)

        if names == None:
        	names = np.zeros(len(vs))

        fig,ax = plt.subplots(len(vs),1,figsize = (10,20))
        for i,v in enumerate(vs):
        	coeffs = np.dot(self.multipole_expansions,v)
        	ax[i].bar(range(Nmulti),coeffs)
        	ax[i].set_title(names[i])
        plt.xticks(range(Nmulti), multipole_names, rotation = -90)
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









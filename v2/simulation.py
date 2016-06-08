"""
simulation.py

Container class for post-processing
resuls from BEMSolver simulations.

"""

import numpy as np
from expansion import spher_harm_expansion

class simulation():

    def __init__(scale):
        self.scale = scale

    
    def import_data(self, path, pointsPerAxis, numElectrodes, saveFile = None):
        '''
        path: file path to the BEMSolver output data
        pointsPerAxis: number of data points in each axis of the simulation grid
        numElectrodes: number of electrodes in the simulation
        saveFile: optional file to save the output dictionary
        
        Import the data for each electrode.
        Generate a three-dimensional array of the potential
        due to each electrode
        
        Returns: TreeDict containing the X, Y, Z axes
        as well as the potential grids for each electrode
        '''

        data = np.loadtxt(path, delimiter=',')
        self.electrode_potentials = []
        
        # get the axis ranges out of the bemsolver data
        X,Y,Z = [0],[0],data[0:pointsPerAxis,2]
        for i in range(0,(pointsPerAxis)):
            if i==0:
                X[0]=(data[pointsPerAxis**2*i+1,0])
                Y[0]=(data[pointsPerAxis*i+1,1])
            else:
                X.append(data[pointsPerAxis**2*i+1,0])
                Y.append(data[pointsPerAxis*i+1,1])
        X,Y = np.array(X).T,np.array(Y).T
        XY = np.vstack((X,Y))
        coord=np.vstack((XY,Z))
        coord=coord.T
        X,Y,Z = coord[:,0],coord[:,1],coord[:,2]
        self.X, self.Y, self.Z = X, Y, Z
        self.numElectrodes = numElectrodes

        # buld the 3D array of potentials for each electrode
        for el in range(numElectrodes):
            n = pointsPerAxis
            self.electrode_potentials.append( np.zeros((n, n, n)) )
            for i in range(n):
                for j in range(n):
                    start_index = n**3*(el) + n**2*i + n*j # start index for this column of data
                    self.electrode_potentials[el][i,j,:] = data[start_index:start_index + n, 3]
        if saveFile is not None:
            fi = open(saveFile, 'wb')
            pickle.dump(output, fi)
            fi.close()

    def expand_potentials(self, expansion_point, order=3):
        '''
        Computes a multipole expansion of every electrode
        around the specified value expansion_point.

        expansion_point: list [x0, y0, z0] of location
        to compute the fiel expansion
        order: int, order of the expansion to carry out

        returns: matrix of multipole expansions for each electrodes
        matrix[:, el] = multipole expansion vector for electrode el
        
        Also defines the class variable self.multipole_expansions
        '''

        N = (order + 1)**2 # number of multipoles

        self.multipole_expansions = np.zeros((N, self.numElectrodes))
        self.expansion_order = order

        X, Y, Z = self.X, self.Y, self.Z

        for el in range(self.numElectrodes):
            potential_grid = self.electrode_potentials[el]
            vec = spher_harm_expansion(potential_grid, expansion_point, X, Y, Z, order)
            self.multipole_expansions[:, el] = vec[0:N].T

        return self.multipole_expansions

    def set_controlled_electrodes(self, controlled_electrodes, shorted_electrodes = []):
        '''
        Define the set of electrodes under DC control

        Arguments:
        controlled_electrodes: list of integers specifying
        the electrodes to be controlled, in the appropriate
        order for the control matrix

        shorted_electrodes: optional. list of electrodes
        shorted together. Form: [(a, b), (c, d, e), ...]

        If some electrodes are shorted, only use one of each
        set in controlled_electrodes.
        '''

        # First adjust the multipole_expansions matrix
        # to account for electrodes shorted together
        # To do this, add the multipole vector for each
        # shorted electrode together
        M_shorted = self.multipole_expansions.copy()
        N = M_shorted[:,0].shape[0] # length of the multipole expansion vector
        for s in shorted_electrodes:
            vec = np.zeros(N)
            for el in s:
                vec += self.multipole_expansions[:, el]
            [M_shorted[:, el] = vec for el in s]

        # multipole expansion matrix after accounting for shorted electrodes
        # and uncontrolled electrodes
        self.reduced_multipole_expansions = np.zeros((N, len(controlled_electrodes)))
        for k, el in enumerate(controlled_electrodes):
            self.reduced_multipole_expansions[:, k] = M_shorted[:,el]

    def generate_control_matrix(self, controlled_multipoles):
        '''
        Generates the multipole control matrix

        controlled_multipoles: list of integers
        specifying which multipoles are to be controlled
        0, 1, 2 correspond to Ex, Ey, Ez
        3-8 correspond to to the quadrupoles
        
        '''

        multipoles = np.zeros((self.expansion_order+1)**2)
        for k in controlled_multipoles:
            multipoles[k] = 1

        

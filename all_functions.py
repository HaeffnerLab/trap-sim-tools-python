"""This is all functions and scripts used by the simulation. All relevant abstraction in project_parameters and analyze_trap."""
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
from treedict import TreeDict
import all_functions

# Primary Functions 
def test_fourth():
    """Construct a text files representing BEM-solver trap simulations.
    This creates 8 electrodes, covering RF, constant, E fields, z**2, and z**4 multipoles."""
    from project_parameters import simulationDirectory,debug
    low,high = -5,6 # defines data with i,j,k=(-2,-1,0,1,2), symmetric for saddle point at origin
    electrode_count = 8 # number of DC electrodes, including the one equivalent to the RF
    axis_length = high-low # number of points along each axis
    electrode_size = axis_length**3 # points per electrode
    e = electrode_size
    total = electrode_count*electrode_size # the length of each simulation data structure
    data = np.zeros((4,total)) # 4 would be 7 instead if we had electric field data
    elem = 0 # initialize the count of data points
    fp = np.sqrt(1/(4*np.pi))
    for i in range(low,high): 
        for j in range(low,high):
            for k in range(low,high):
                r = np.sqrt(i**2+j**2+k**2)
                data[:,elem]      = [i,j,k,fp*np.sqrt(15)*0.5*(i**2-j**2)] # DC_0 aka RF
                data[:,elem+e]    = [i,j,k,fp] # DC_1, appears to be the same for fp=1
                data[:,elem+2*e]  = [i,j,k,fp*np.sqrt(3)*i] # DC_2
                data[:,elem+3*e]  = [i,j,k,fp*np.sqrt(3)*j] # DC_3
                data[:,elem+4*e]  = [i,j,k,fp*np.sqrt(3)*k] # DC_4
                data[:,elem+5*e]  = [i,j,k,fp*np.sqrt(5)*(k**2-0.5*(i**2+j**2))] # DC_5
                data[:,elem+6*e]  = [i,j,k,fp*np.sqrt(7)*(0.5*k*(2*k**2-3*i**2-3*j**2))] # DC_6
                data[:,elem+7*e]  = [i,j,k,fp*np.sqrt(9)*(35*k**4-30*(k*r)**2+3*r**4)/8] # DC_7
                elem += 1
        np.savetxt('{0}fourth1.txt'.format(simulationDirectory),data.T,delimiter=',')
    return 'test fourth constructed'

def test_text():
    """Construct a pair of text files representing BEM-solver trap simulations.
    This makes the primary synthetic data structure for testing."""
    from project_parameters import simulationDirectory,debug
    low,high = -2,3 # defines data with i,j,k=(-2,-1,0,1,2), symmetric for saddle point at origin
    electrode_count = 14 # number of DC electrodes, including the one equivalent to the RF
    if debug.calibrate:
        electrode_count= 9
    axis_length = high-low # number of points along each axis
    electrode_size = axis_length**3 # points per electrode
    e = electrode_size
    total = electrode_count*electrode_size # the length of each simulation data structure
    for sim in range(1,3): # construct 2 simulations 
        data = np.zeros((4,total)) # 4 would be 7 instead if we had electric field data
        elem = 0 # initialize the count of data points
        for i in range(low,high): 
            for j in range(low,high):
                if sim == 1:
                    zlow,zhigh = -4,1
                if sim == 2:
                    zlow,zhigh = 0,5
                for k in range(zlow,zhigh):
                    # there is no DC_0 and the final DC is also the RF
                    Ex,Ey,Ez = i,j,k
                    #Ex,Ey,Ez = i,-k,j
                    U1,U2,U3,U4,U5 = 0.5*(i**2-j**2),0.5*(2*k**2-i**2-j**2),i*j,k*j,i*k
                    RF = U1+10**-3*(j**3-3*i**2*j)
                    # assign data points to each electrode
                    if debug.calibrate:
                        data[:,elem]      = [i,j,k,RF] # DC_0 aka RF
                        data[:,elem+e]    = [i,j,k,Ex] # DC_1
                        data[:,elem+2*e]  = [i,j,k,Ey] # DC_2
                        data[:,elem+3*e]  = [i,j,k,Ez] # DC_3
                        data[:,elem+4*e]  = [i,j,k,U1] # DC_4
                        data[:,elem+5*e]  = [i,j,k,U2] # DC_5
                        data[:,elem+6*e]  = [i,j,k,U3] # DC_6
                        data[:,elem+7*e]  = [i,j,k,U4] # DC_7
                        data[:,elem+8*e]  = [i,j,k,U5] # DC_8
                            
                    else:    
                        data[:,elem]      = [i,j,k,RF]    # DC_0 aka RF
                        data[:,elem+e]    = [i,j,k,Ex]    # DC_1
                        data[:,elem+2*e]  = [i,j,k,Ey]    # DC_2
                        data[:,elem+3*e]  = [i,j,k,Ez]    # DC_3
                        data[:,elem+4*e]  = [i,j,k,U1+U2] # DC_4
                        data[:,elem+5*e]  = [i,j,k,U1+U3] # DC_5
                        data[:,elem+6*e]  = [i,j,k,U1+U4] # DC_6
                        data[:,elem+7*e]  = [i,j,k,U1+U5] # DC_7
                        data[:,elem+8*e]  = [i,j,k,U2+U3] # DC_8
                        data[:,elem+9*e]  = [i,j,k,U2+U4] # DC_9
                        data[:,elem+10*e] = [i,j,k,U2+U5] # DC_10
                        data[:,elem+11*e] = [i,j,k,U3+U4] # DC_11
                        data[:,elem+12*e] = [i,j,k,U3+U5] # DC_12
                        data[:,elem+13*e] = [i,j,k,U4+U5] # DC_13
                    elem += 1
        if debug.calibrate:
           np.savetxt('{0}meshless-pt{1}.txt'.format(simulationDirectory,sim),data.T,delimiter=',') 
        else:
            np.savetxt('{0}synthetic-pt{1}.txt'.format(simulationDirectory,sim),data.T,delimiter=',')
    return 'test_text constructed, 2 simulations'

def import_data():
    """Originally created as importd by Mike and modified by Gebhard, Oct 2010.
    Conventions redefined, cleaned up, and combined with new developments by Nikos, Jun 2013.
    Converted to Python and simplified by William, Jan 2014.
    Imports simulation data stored in text files, interpreting three coordinates, a scalar potential value, 
    and (optionally) three electric field components. It outputs a Python "pickle" for each text file.
    Each pickle contains attributes: potentials (with an attribute for each electrode), grid vectors (X,Y,Z), 
    and system Information from project_parameters.     
    * All the electrodes are initially assumed to be DC.
    * The sequence for counting DC electrodes runs through the left side of the RF (bottom to top), right side of
    the RF (bottom to top), center electrodes inside of the RF (left center, then right center), and finally RF.
    *All other conventions are defined and described in project_parameters."""
    from project_parameters import simCount,perm,dataPointsPerAxis,numElectrodes,save,debug,scale
    from project_parameters import baseDataName,simulationDirectory,fileName,savePath,timeNow,useDate
    import pickle
    # renaming for convenience
    na, ne = dataPointsPerAxis, numElectrodes 
    [startingSimulation,numSimulations] = simCount
    # iterate through each simulation text file
    for iterationNumber in range(startingSimulation,startingSimulation+numSimulations):        
        #########################################################################################
        #0) Check if data already exists 
        def fileCheck(iterationNumber):
            """Helper function to determine if there already exists imported data."""
            try:
                file = open(savePath+fileName+'_simulation_{}'.format(iterationNumber)+'.pkl','rb')
                file.close()
                if iterationNumber==numSimulations+1:
                    return 'done'
                return fileCheck(iterationNumber+1)
            except IOError: # unable to open pickle because it does not exist
                print('No pre-imported data in directory for simulation {}.'.format(iterationNumber))
                return iterationNumber
        iterationNumber=fileCheck(iterationNumber) # lowest iteration number that is not yet imported
        if iterationNumber=='done':
            return 'All files have been imported.'
        #########################################################################################
        # Read txt file
        print('Importing '+''+baseDataName+str(iterationNumber)+'...')
        dataName=(simulationDirectory+baseDataName+str(iterationNumber)+'.txt')
        #1) check if there is BEM-solver data to import
        try: 
            DataFromTxt=np.loadtxt(dataName,delimiter=',') 
        except IOError:
            return ('No BEM-solver data to import for simulation {}. Import complete.'.format(iterationNumber))
        #2) build the X,Y,Z grids
        X,Y,Z = [0],[0],DataFromTxt[0:na,2]
        for i in range(0,(na)):
            if i==0:
                X[0]=(DataFromTxt[na**2*i+1,0])
                Y[0]=(DataFromTxt[na*i+1,1])
            else:
                X.append(DataFromTxt[na**2*i+1,0])
                Y.append(DataFromTxt[na*i+1,1])
        X,Y = np.array(X).T,np.array(Y).T
        XY = np.vstack((X,Y))
        coord=np.vstack((XY,Z))
        coord=coord.T
        X,Y,Z = coord[:,perm[0]]/scale,coord[:,perm[1]]/scale,coord[:,perm[2]]/scale
        if debug.import_data:
            print ('Printing grid vectors X,Y, and Z:')
            print (X,Y,Z)
        #3) load all the voltages and E vector into struct using dynamic naming 
        struct=TreeDict() # begin intermediate shorthand.
        for el in range(ne): #el refers to the electrode, +1 is to include EL_0, the RF electrode
            struct['EL_DC_{}'.format(el)]=np.zeros((na,na,na))
            for i in range(na):
                for j in range (na):
                    lb = na**3*(el) + na**2*i + na*j # lower bound defined by electrodes complete and axes passed
                    ub = lb + na # upper bound defined by an increase by axis length
                    struct['EL_DC_{}'.format(el)][i,j,:]=DataFromTxt[lb:ub,3]
            struct['EL_DC_{}'.format(el)]=np.transpose(struct['EL_DC_{}'.format(el)],perm)
        del DataFromTxt
        #4) Build the simulation data structure
        sim=struct                                # copy over intermediate dynamic data structure
        sim.X,sim.Y,sim.Z=X,Y,Z                   # set grid vectors
        sim.EL_RF = struct.EL_DC_0                # set RF potential field
        s = sim                           # shorthand for defining import configuration branch
        s.simulationDirectory = simulationDirectory
        s.baseDataName = baseDataName
        s.timeNow = timeNow
        s.fileName = fileName
        s.useDate = useDate
        s.simCount = simCount
        s.dataPointsPerAxis = na
        s.numElectrodes = ne
        s.savePath = savePath
        s.perm = perm
        #sim.systemInformation = s
        if debug.import_data: # Plot each electrode
            print(plot_potential(sim.EL_RF,X,Y,Z,'1D plots','Debug: RF electrode'))
            for el in range(1,ne):                
                print(plot_potential(sim['EL_DC_{}'.format(el)],X,Y,Z,'1D plots','Debug: DC electrode {}'.format(el)))
        #5) save the particular simulation as a pickle data structure
        if save == True:
            name=savePath+fileName+'_simulation_{}'.format(iterationNumber)+'.pkl'
            print ('Saving '+name+' as a data structure...')
            output = open(name,'wb')
            pickle.dump(sim,output)
            output.close()
    return 'Import complete.'
     
def get_trap():
    """Originally getthedata.
    Create a new "trap" structure centered around the given position and composed of portions of the adjacent simulations.
    The electrodes are ordered as E[1]=E[DC_1],...,E[max-1]=E[center],E[max]=E[RF].
    Connect together a line of cubic matrices to describe a rectangular prism of data.
    The consecutive data structures must have overlaping first and last points. 
    Used to create field configuration attributes on the trap that will be used by lower order functions. 
    These are now imported and saved wherever they are first used through the remaining higher order functions.
    Recall that the grid vectors X,Y,Z are still attributes of potentials now and will become attributes of instance later.
    Created by Nikos 2009, cleaned 26-05-2013, 10-23-2013.
    Converted to Python and revised by William Jan 2014"""
    #0) define parameters
    from project_parameters import fileName,savePath,position,zMin,zMax,zStep,save,debug,name,simCount,charge,mass,r0
    import pickle
    tf=TreeDict() # begin shorthand for trap data structure
    pathName = savePath+fileName+'_simulation_'
    #1) Check if the number of overlapping data structures is the same as the number of simulations.
    # simCount was imported again (after import_data used it) because it is used as a check for user input consistency
    numSim=int(np.ceil(float(zMax-zMin)/zStep-1e-9))
    if numSim!=simCount[1]:
        print numSim,simCount,float(zMax-zMin)/zStep
        raise Exception('Inconsistency in simulation number. Check project_parameters for consistency.')
    if numSim==1:
        print('If there is only one simulation, use that one. Same debug as import_data.')
        # This is redundant with the final save but is called here to avoid errors being raised with zLim below.
        file = open(pathName+str(simCount[0])+'.pkl','rb')
        tf.potentials = pickle.load(file)
        file.close()
        potential=tf.potentials    # base potential to write over
        ne=potential.numElectrodes
        trap = tf
        c=trap.configuration
        c.position = position 
        c.numElectrodes = ne # also listed in systemInformation
        c.charge = charge
        c.mass = mass
        c.r0 = r0
        trap.configuration=c
        if save:
            import pickle
            name=savePath+name+'.pkl'
            print('Saving '+name+' as a data structure...')
            output = open(name,'wb')
            pickle.dump(trap,output)
            output.close()
        return 'Constructed trap from single file.'
    #2) Define a background for files. 
    zLim=np.arange(zMin,zMax,zStep) 
    #3) helper function to find z-position of ion
    def find_index(list,position): # Find which simulation position is in based on z-axis values.
        """Finds index of first element of list higher than a given position. Lowest index is 1, not 0"""
        # replaces Matlab Code: I=find(zLim>position,1,'first')-1
        i=0
        for elem in list:
            if elem>position:
                index=i
                return index
            else: 
                index=0
            i += 1
        return index 
    index=find_index(zLim,position)
    if (index<1) or (index>simCount[1]):
        raise Exception('Invalid ion position. Quitting.')
    #4) determine which side of the simulation the ion is on
    pre_sign=2*position-zLim[index-1]-zLim[index] # determines 
    if pre_sign==0: # position is exactly halfway between simulations
        sign=-1 # could be 1 as well
    else:
        sign=int(pre_sign/abs(pre_sign))  
    #5) If position is in the first half of the first grid, just use that grid.
    if (index==1) and (sign==-1): 
        print('If position is in the first or last grid, just use that grid.')
        file = open(pathName+'1.pkl','rb')
        tf.potentials = pickle.load(file)
        file.close()
    #6) If the ion is in the second half of the last grid, just use the last grid. 
    elif (index==simCount[1]) and (sign==1): 
        print('If the ion is in the second half of the last grid, just use the last grid.')
        file = open(pathName+'{}.pkl'.format(numSimulations),'rb')
        tf.potentials = pickle.load(file)
        file.close()
    #7) Somewhere in between. Build a new set of electrode potentials centered on the position.
    else:
        print('Somewhere in between. Build a new set of electrode potentials centered on the position.')
        #a) open data structure
        file = open(pathName+'{}.pkl'.format(index),'rb')
        tf.potentials = pickle.load(file)
        file.close()
        lower=position-zStep/2 # lower bound z value 
        upper=position+zStep/2 # upper bound z value
        shift=int(pre_sign)    # index to start from in left sim and end on in right sim, how many idices the indexing shifts right by
        if shift < 0:
            index -= 1
            shift = abs(shift)
        #b) open left sim
        file1 = open(pathName+'{}.pkl'.format(index),'rb')
        left = pickle.load(file1)
        file1.close()
        #c) open right sim
        file2 = open(pathName+'{}.pkl'.format(index+1),'rb')
        right = pickle.load(file2)
        file.close()
        #d) build bases
        cube=tf.potentials.EL_DC_1 # arbitrary electrode; assume each is cube of same length
        w=len(cube[0,0,:])         # number of elements in each cube; width 
        potential=tf.potentials    # base potential to write over
        Z=potential.Z              # arbitrary axis with correct length to build new grid vector
        ne=potential.numElectrodes
        #e) build up trap
        for el in range(ne): # includes the RF
            right_index,left_index,z_index=w-shift,0,0
            temp=np.zeros((w,w,w)) # placeholder that becomes each new electrode
            left_el=left['EL_DC_{}'.format(el)]
            right_el=right['EL_DC_{}'.format(el)]
            for z in range(shift-1,w): # build up the left side
                temp[:,:,left_index]=left_el[:,:,z]
                Z[z_index] = left.Z[z]
                z_index += 1
                left_index += 1
            z_index -= 1 # counters double-counting of overlapping center point
            for z in range(shift): # build up the right side; ub to include final point
                temp[:,:,right_index]=right_el[:,:,z]
                Z[z_index] = right.Z[z]
                z_index += 1
                right_index+=1
            potential.Z = Z
            potential['EL_DC_{}'.format(el)]=temp
        tf.potentials = potential
    #8) assign configuration variables to trap; originally trapConfiguration
    trap = tf
    c=trap.configuration
    c.position = position 
    c.numElectrodes = ne # also listed in systemInformation
    c.charge = charge
    c.mass = mass
    c.r0 = r0
    trap.configuration=c
    #9) check if field generated successfully
    if debug.get_trap:
        sim = tf.potentials
        X,Y,Z = sim.X,sim.Y,sim.Z # grid vectors  
        plt.plot(Z,np.arange(len(Z)))
        plt.title('get_trap: contnuous straight line if successful')
        plt.show()  
        sim = tf.potentials
        print(plot_potential(sim.EL_RF,X,Y,Z,'2D plots','Debug: RF electrode'))
        for el in range(1,ne):               
            print(plot_potential(sim['EL_DC_{}'.format(el)],X,Y,Z,'1D plots','Debug: DC electrode {}'.format(el)))
    #10) save new data structure as a pickle    
    if save:
        import pickle
        name=savePath+name+'.pkl'
        print('Saving '+name+' as a data structure...')
        output = open(name,'wb')
        pickle.dump(trap,output)
        output.close()
    return 'Constructed trap.'

def expand_field():
    """Originally regenthedata. Regenerates the potential data for all electrodes using multipole expansion to given order.
    Also returns a attribute of trap, configuration.multipoleCoefficients, which contains the multipole coefficients for all electrodes.
    The electrodes are ordered as E[1], ..., E[NUM_DC]=E[RF] though the final electrode is not included in the attribute.
    6/19/14: RF is not 0, not NUM_DC
    (if center and RF are used)
          ( multipoles    electrodes ->       )
          (     |                             )
    M =   (     V                             )
          (                                   )
    Multipole coefficients only up to order 8 are kept, but the coefficients are calculated up to order L.
    trap is the path to a data structure that contains an instance with the following properties
    .DC is a 3D matrix containing an electric potential and must solve Laplace's equation
    .X,.Y,.Z are the vectors that define the grid in three directions
    Xcorrection, Ycorrection: optional correction offsets from the RF saddle point,
                              in case that was wrong by some known offset
    position: the axial position where the ion sits
    Written by Nikos, Jun 2009, cleaned up 26-05-2013, 10-23-2013
    Converted to by Python by William, Jan 2014"""
    #0) establish parameters and open updated trap with including instance configuration attributes
    from project_parameters import savePath,name,Xcorrection,Ycorrection,regenOrder,save,debug,E,pureMultipoles
    #from all_functions import spher_harm_bas,spher_harm_exp,spher_harm_cmp,spher_harm_qlt,find_saddle,exact_saddle,plot_potential,dc_potential
    import pickle
    trap = savePath+name+'.pkl'
    file = open(trap,'rb')
    tf = pickle.load(file)
    file.close()
    ne=tf.configuration.numElectrodes
    if not debug.expand_field:
        if tf.configuration.expand_field==True:
            return 'Field is already expanded.'
    VMULT= np.ones((ne,1)) # analogous to dcVolatages
    dc_potential(trap,VMULT,E,True) # run dc_potential to create instance configuration
    file = open(trap,'rb')
    tf = pickle.load(file)
    file.close()
    V,X,Y,Z=tf.instance.DC,tf.instance.X,tf.instance.Y,tf.instance.Z
    tc=tf.configuration #intermediate shorthand for configuration
    position = tc.position
    tc.EL_RF = tf.potentials.EL_RF
    if Xcorrection:
        print('expand_field: Correction of XRF: {} mm.'.format(str(Xcorrection)))
    if Ycorrection:
        print('expand_field: Correction of YRF: {} mm.'.format(str(Ycorrection)))
    # Order to expand to in spherharm for each electrode.
    NUM_DC = ne # includes RF as DC_0
    order = np.zeros(NUM_DC)
    L = regenOrder
    order[:]=int(L)
    N=(L+1)**2 
    #1) Expand the RF about the grid center, regenerate data from the expansion.
    print('Building RF potential')
    Irf,Jrf,Krf = int(np.floor(X.shape[0]/2)),int(np.floor(Y.shape[0]/2)),int(np.floor(Z.shape[0]/2))
    Xrf,Yrf,Zrf = X[Irf],Y[Jrf],Z[Krf]
    rfbasis,rfscale = spher_harm_bas(Xrf,Yrf,Zrf,X,Y,Z,L)
    Qrf = spher_harm_exp(tc.EL_RF,rfbasis,rfscale)
    if debug.expand_field: 
       plot_potential(tc.EL_RF,X,Y,Z,'1D plots','EL_RF','V (Volt)',[Irf,Jrf,Krf])
    print('Comparing RF potential')
    tc.EL_RF = spher_harm_cmp(Qrf,rfbasis,rfscale,L)
    if debug.expand_field: 
        plot_potential(tc.EL_RF,X,Y,Z,'1D plots','EL_RF','V (Volt)',[Irf,Jrf,Krf])
    #2) Expand the RF about its saddle point at the trapping position and save the quadrupole components.
    print('Determining RF saddle')
    [Xrf,Yrf,Zrf] = exact_saddle(tc.EL_RF,X,Y,Z,2,position) 
    [Irf,Jrf,Krf] = find_saddle(tc.EL_RF,X,Y,Z,2,position) 
    print('Building DC Basis')
    dcbasis,dcscale = spher_harm_bas(Xrf+Xcorrection,Yrf+Ycorrection,Zrf,X,Y,Z,int(order[0]))
    Qrf = spher_harm_exp(tc.EL_RF,dcbasis,dcscale)  
    tc.Qrf = 2*[Qrf[7][0]*3,Qrf[4][0]/2,Qrf[8][0]*6,-Qrf[6][0]*3,-Qrf[5][0]*3]
    tc.thetaRF = 45*((Qrf[8][0]/abs(Qrf[8][0])))-90*np.arctan((3*Qrf[7][0])/(3*Qrf[8][0]))/np.pi
    #3) Regenerate each DC electrode
    Mt=np.zeros((N,NUM_DC)) 
    for el in range(0,NUM_DC): # Expand all the electrodes and  regenerate the potentials from the multipole coefficients; include RF
        print('Expanding DC Electrode {} ...'.format(el))        
        multipoleDCVoltages = np.zeros(NUM_DC)
        multipoleDCVoltages[el] = 1 
        E = [0,0,0]
        Vdc = dc_potential(trap,multipoleDCVoltages,E) 
        if debug.expand_field:
            plot_potential(Vdc,X,Y,Z,'1D plots',('Old EL_{} DC Potential'.format(el)),'V (Volt)',[Irf,Jrf,Krf])
            # Vdc += 0*Vdc-Vdc[Irf,Jrf,Krf]
            # plot_potential(Vdc,X,Y,Z,'1D plots',('Shifted EL_{} DC Potential'.format(el)),'V (Volt)',[Irf,Jrf,Krf])
        Q = spher_harm_exp(Vdc,dcbasis,dcscale) 
        print('Regenerating Electrode {} potential...'.format(el))
        Mi,Mj = Q.copy(),Q.copy() # intermediate, will be rescaled for plotting in spher_harm_cmp
        tf.potentials['EL_DC_{}'.format(el)]=spher_harm_cmp(Mi,dcbasis,dcscale,int(order[el]))
        if debug.expand_field:
            print Q[0:9]
            plot_potential(tf.potentials['EL_DC_{}'.format(el)],X,Y,Z,'1D plots',('EL_{} DC Potential'.format(el)),'V (Volt)',[Irf,Jrf,Krf])
        Q = Mj   
        Mt[:,el] = Q[0:N].T  
    #4) Define the multipole Coefficients
    tc.multipoleCoefficients = Mt
    print('expand_field: Size of the multipole coefficient matrix is {}'.format(Mt.shape))
    if save: 
        tc.expand_field=True
    tf.configuration=tc
    dataout=tf
    if save: 
        output = open(trap,'wb')
        pickle.dump(tf,output)
        output.close()
    return 'expand_field: ended successfully.'
 
def trap_knobs():
    """Updates trap.configuration with the matrix which controls the independent multipoles, and the kernel matrix. 
    Starting from the matrix multipoleCoefficients, return a field multipoleControl with
    the linear combimations of trap electrode voltages that give 1 V/mm, or 1 V/mm**2 of the multipole number i.
    Also return matrix multipoleKernel which is the kernel matrix of electrode linear combinations which do nothing to the multipoles.
    The order of multipole coefficients is:
    1/r0**[ x, y, z ] and 
    1/r0**2*[ (x^2-y^2)/2, (2z^2-x^2-y^2)/2, xy, yz, xz ], where r0 is 1 mm (unless rescaling is applied)
    Before solving the system, compact the multipoleCoefficient matrix by removing all redundant electrodes.
    After solving the system, expand the multipoleControl matric to include these.
    If the system is underdetermined, then there is no Kernel or regularization."""
    print('Executing trap_knobs...')
    #0) Define parameters and heck to see what scripts have been run
    from project_parameters import save,debug,trapFile,elMap,electrodes,multipoles,name,simulationDirectory,reg
    #from all_functions import nullspace,plotN
    import pickle
    file = open(trapFile,'rb')
    trap = pickle.load(file)
    file.close()
    if trap.configuration.expand_field!=True:
        return 'You must run expand_field first!'
    if trap.configuration.trap_knobs and not debug.trap_knobs:
        return 'Already executed trap_knobs.'
    #1) redefine parameters with shorthand and run sanity checks
    totE = len(electrodes) # numTotalElectrodes
    totM = len(multipoles) # numTotalMultipoles
    assert totE == trap.configuration.numElectrodes # Make sure that the total number of electrodes includes the RF.
    tc = trap.configuration
    mc = tc.multipoleCoefficients # this is the original, maximum-length multipole coefficients matrix (multipoles,electrodes)
    for row in range(totM):
        row+=1
        if abs(np.sum(mc[row,:])) < 10**-50: # arbitrarily small
            return 'trap_knobs: row {} is all 0, can not solve least square, stopping trap knobs'.format(row)
    #2) Apply electrode mapping by clearing some electrodes and adding them to the new map
    # mapping one to an unused electrode should turn it off as well
    for index in range(totE):
        if index != elMap[index]:
            mc[:,elMap[index]] += mc[:,index] # combine the electrode to its mapping
            electrodes[index] = 0 # clear the mapped electrode, implemented in part 3
    useE = int(np.sum(electrodes)) # numUsedElectrodes
    useM = int(np.sum(multipoles)) # numUsedMultipoles
    eo = np.sqrt(useM)-1 # expansionOrder
    #3) build a reduced array of multipole coefficients to invert
    MC = np.zeros((useM,useE)) # reduced matrix to build up and invert
    ML = 0
    for ml in range(totM):
        if multipoles[ml]:
            EL = 0 #  clear the electrode indexing before looping through electrodes again for new multipole
            for el in range(totE):
                if electrodes[el]:
                    MC[ML,EL] = mc[ml,el]
                    EL += 1
            ML += 1
    print('trap_knobs: with electrode and multipole constraints, the coefficient matrix size is ({0},{1}).'.format(MC.shape[0],MC.shape[1]))
    #4) numerially invert the multipole coefficients to get the multipole controls, one multipole at a time
    # solve the equation MC*A = B, where the matrix made from all A vectors is C
    C = np.zeros((useE,useM)) # multipole controls (electrodes,multipoles) will be the inverse of multipole coefficents 
    for mult in range(useM):
        B = np.zeros(useM)
        B[mult] = 1
        A = np.linalg.lstsq(MC,B)[0]
        C[:,mult] = A
    #5) calculate the nullspace and regularize if the coefficients matrix is sufficiently overdetermined
    if useM < useE:
        K = nullspace(MC)
    else:
        print('There is no nullspace because the coefficient matrix is rank deficient.\nThere can be no regularization.')
        K = None
        reg = False
    if reg:
        for mult in range(useM):
            Cv = C[:,mult].T
            Lambda = np.linalg.lstsq(K,Cv)[0]
            test=np.dot(K,Lambda)
            C[:,mult] = C[:,mult]-test      
    #6) expand the matrix back out again, with zero at all unused electrodes and multipoles; same as #3 with everything flipped
    c = np.zeros((totE,totM)) 
    EL = 0
    for el in range(totE):
        if electrodes[el]:
            ML = 0 #  clear the electrode indexing before looping through electrodes again for new multipole
            for ml in range(totM):
                if multipoles[ml]:
                    c[el,ml] = C[EL,ML]
                    ML += 1
            EL += 1
    #7) plot the multipole controls in teh trap geometry
    if debug.trap_knobs:
        for ml in range(totM):
            if multipoles[ml]:
                plot = c[:,ml]
                plotN(plot,'Multipole {}'.format(ml)) 
    #8) update instance configuration with multipole controls to be used bu dc_voltages in post_process_trap
    tc.multipoleKernel = K
    tc.multipoleControl = c
    tc.trap_knobs = True
    trap.configuration=tc
    #8.5) change the order of the columns of c for labrad
    # originally (after constant) Ez,Ey,Ex,U3,U4,U2,U5,U1,...,Y4-4,...,Y40,...,Y44
    # we want it to be Ex,Ey,Ez,U2,U1,U3,U5,U4,...,Y40 and then end before any of the other 4th order terms
#     cc = c.copy()
#     cc[:,1] = c[:,3] # Ex
#     cc[:,2] = c[:,1] # Ey
#     cc[:,3] = c[:,2] # Ez
#     cc[:,4] = c[:,6] # U2
#     cc[:,5] = c[:,8]
#     cc[:,6] = 0 # U3
#     cc[:,16]= c[:,20]# Y40
#     cc[:,20]= 0
    cc = c
    #9) save trap and save c as a text file (called the "C" file for Labrad)
    if save: 
        import pickle
        print('Saving '+name+' as a data structure...')
        output = open(trapFile,'wb')
        pickle.dump(trap,output)
        output.close()
        #ct = cc[1:totE,1:17] #eliminate the RF electrode and constant multipole; eliminate everything past Y40
        ct = cc[1:totE,1:25] # only eliminate the constant
        text = np.zeros((ct.shape[0])*(ct.shape[1]))
        for j in range(ct.shape[1]):
            for i in range(ct.shape[0]):
                text[j*ct.shape[0]+i] = ct[i,j]
        np.savetxt(simulationDirectory+name+'.txt',text,delimiter=',')
    return 'Completed trap_knobs.'
 
def post_process_trap():
    """A post processing tool that analyzes the trap. This is the highest order function.
    It plots an input trap in a manner of ways and returns the frequencies and positions determined by pfit.
    Before 2/19/14, it was far more complicated. See ppt2 for the past version and variable names.
    All necessary configuration parameters should be defined by dcpotential instance, trap knobs, and so on prior to use.
    Change rfplot and dcplot to control the "dim" input to plot_potential for plotting the potential fields.
    There is also an option to run findEfield that determines the stray electric field for given DC voltages.
    As of May 2014, only "justAnalyzeTrap" is in use, so see older versions for the other optimizations.
    This is primarily a plotting function beyond just calling pfit. 
    RF saddles are dim=2 because there ion is not contained along teh z-axis. DC saddles are dim=3.
    Nikos, January 2009
    William Python Feb 2014""" 
    #################### 0) assign internal values #################### 
    from project_parameters import debug,savePath,name,driveAmplitude,driveFrequency,Omega,dcplot,weightElectrodes,coefs,ax,az,phi,save,scale
    #from all_functions import find_saddle,plot_potential,dc_potential,set_voltages,exact_saddle,spher_harm_bas,spher_harm_exp,pfit,plotN
    import pickle    
    trap = savePath+name+'.pkl'
    file = open(trap,'rb')
    tf = pickle.load(file)
    file.close()
    qe = tf.configuration.charge
    mass = tf.configuration.mass
    Zval = tf.configuration.position
    r0 = tf.configuration.r0
    RFampl = driveAmplitude 
    V0 = mass*(2*np.pi*Omega)**2*(r0*10**-3)**2/qe
    X,Y,Z=tf.instance.X,tf.instance.Y,tf.instance.Z    
    data = tf.configuration
    dcVoltages = set_voltages()
    ne = len(weightElectrodes)
    E = tf.instance.E
    out = tf.configuration
    if debug.post_process_trap:
        #print dcVoltages,np.max(dcVoltages)#np.sum(abs(dcVoltages))
        #plotN(dcVoltages,'set DC voltages')
        Vdc = dc_potential(trap,dcVoltages,E)
        [IDC,JDC,KDC] = find_saddle(Vdc,X,Y,Z,3,Zval)        
        #[XDC,YDC,ZDC] = exact_saddle(Vdc,X,Y,Z,3,Zval)
        XDC,YDC,ZDC = X[IDC],150/scale,Z[KDC]
        print XDC,YDC,ZDC,IDC,JDC,KDC
        dcbasis,dcscale= spher_harm_bas(XDC,YDC,ZDC,X,Y,Z,4)
        QQ = spher_harm_exp(Vdc,dcbasis,dcscale) 
        print QQ[0:9].T
    #1) RF Analysis
    print('RF Analysis')           
    Vrf = RFampl*data.EL_RF
    [Irf,Jrf,Krf] = find_saddle(Vrf,X,Y,Z,2,Zval)
    print X[10],Y[12],Z[10]
    if debug.post_process_trap:
        plot_potential(Vrf,X,Y,Z,dcplot,'weighted RF potential','V_{rf} (eV)',[Irf,Jrf,Krf])
    #2) DC Analysis
    print('DC Analysis')
    Vdc = dc_potential(trap,dcVoltages,E,update=None)
    [Idc,Jdc,Kdc] = find_saddle(Vdc,X,Y,Z,3,Zval) # only used to calculate error at end
    if debug.post_process_trap:
        plot_potential(Vdc,X,Y,Z,'1D plots','full DC potential')
    #3) determine the exact saddles of the RF and DC
    Vdc = dc_potential(trap,dcVoltages,E)
    print('Determining exact RF saddle...')
    [Xrf,Yrf,Zrf] = exact_saddle(Vrf,X,Y,Z,2,Zval)  
    print('Determining exact DC saddle...')
    [Xdc,Ydc,Zdc] = exact_saddle(Vdc,X,Y,Z,3,Zval)
    #4) determine stray field (beginning of justAnalyzeTrap)
    print('Determining compensation due to E field...')
    nx,ny,nz=X.shape[0],Y.shape[0],Z.shape[0]
    x,y,z = np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x[i,j,k] = X[i]
                y[i,j,k] = Y[j]
                z[i,j,k] = Z[k]
    VlessE = Vdc-E[0]*x-E[1]*y-E[2]*z
    [Xdc,Ydc,Zdc] = exact_saddle(VlessE,X,Y,Z,3) 
    dist = np.sqrt((Xrf-Xdc)**2+(Yrf-Ydc)**2+(Zrf-Zdc)**2)  
    #5) call pfit to built teh total field and determine the trap characteristics
    [fx,fy,fz,theta,Depth,Xe,Ye,Ze] = pfit(Vrf,Vdc,X,Y,Z,Irf,Jrf,Krf)#pfit(trap,E,Freq,RFampl)
    print('Stray field is ({0},{1},{2}) V/m.'.format(scale*E[0],scale*E[1],scale*E[2]))
    print('With this field, the compensation is optimized to {} micron.'.format(scale*dist))
    print Irf,Jrf,Krf,Idc,Jdc,Kdc
    print('RF saddle: ({0},{1},{2})\nDC saddle ({3},{4},{5}).'.format(Xrf,Yrf,Zrf,Xdc,Ydc,Zdc)) 
    if debug.trap_depth:
        print('The trap escape position is at ({0},{1},{2}) microns, for a trap depth of {3} mV'.format(Xe*scale,Ye*scale,Ze*scale,Depth*scale))
    print('The trap frequencies are fx = {0} MHz, fy = {1} MHz, and fz = {2} MHz'.format(fx*10**-6,fy*10**-6,fz*10**-6))
    #6) Sanity testing; quality check no longer used
    if debug.post_process_trap:
        rfbasis,rfscale= spher_harm_bas(Xrf,Yrf,Zrf,X,Y,Z,2)
        Qrf = spher_harm_exp(Vrf,rfbasis,rfscale)           
        if np.sqrt((Xrf-Xdc)**2+(Yrf-Ydc)**2+(Zrf-Zdc)**2)>0.008: 
            print('Expanding DC with RF for saniy checking.')
            Qdc = spher_harm_exp(Vdc,rfbasis,rfscale) 
        else:
            print('Expanding DC without RF for sanity checking.')
            dcbasis,dcscale= spher_harm_bas(Xdc,Ydc,Zdc,X,Y,Z,2)
            Qdc = spher_harm_exp(Vdc,dcbasis,dcscale) 
        Arf = 2*np.sqrt( (3*Qrf[7])**2+(3*Qrf[8])**2 )
        Thetarf = 45*(Qrf[8]/abs(Qrf[8]))-90*np.arctan((3*Qrf[7])/(3*Qrf[8]))/np.pi
        Adc = 2*np.sqrt( (3*Qdc[7])**2+(3*Qdc[8])**2 )
        Thetadc = 45*(Qrf[8]/abs(Qrf[8]))-90*np.arctan((3*Qdc[7])/(3*Qdc[8]))/np.pi
        out.E = E
        out.miscompensation = dist
        out.ionpos = [Xrf,Yrf,Zdc]
        out.ionposIndex = [Irf,Jrf,Krf]
        out.frequency = [fx,fy,fz]
        out.theta = theta
        out.trap_depth = Depth/qe 
        out.escapepos = [Xe,Ye,Ze]
        out.Quadrf = 2*np.array([Qrf[7]*3,Qrf[4]/2,Qrf[8]*6,-Qrf[6]*3,-Qrf[5]*3])
        out.Quaddc = 2*np.array([Qdc[7]*3,Qdc[4]/2,Qdc[8]*6,-Qdc[6]*3,-Qdc[5]*3])
        out.Arf = Arf
        out.Thetarf = Thetarf
        out.Adc = Adc
        out.Thetadc = Thetadc
        T = np.array([[2,-2,0,0,0],[-2,-2,0,0,0],[0,4,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0, 0,0,0,1]])
        Qdrf = out.Quadrf.T
        Qddc = out.Quaddc.T
        out.q = (1/V0)*T*Qdrf
        out.alpha = (2/V0)*T*Qddc
        out.Error = [X[Idc]-Xdc,Y[Jdc]-Ydc,Z[Kdc]-Zdc]
    #7) update the trapping field data structure with instance attributes
    tf.configuration=out
    tf.instance.driveAmplitude = driveAmplitude
    tf.instance.driveFrequency = driveFrequency
    tf.instance.coefs = coefs
    tf.instance.ax = ax
    tf.instance.az = az
    tf.instance.phi = phi
    tf.instance.ppt = True
    tf.instance.out = out
    if save==True:
        import pickle
        update=trap
        print('Saving '+update+' as a data structure...')
        output = open(update,'wb')
        pickle.dump(tf,output)
        output.close()
    return 'post_proccess_trap complete' #out # no output needed really

print('Referencing all_functions...')
# Secondary Functions
def dc_potential(trap,VMULT,E,update=None):
    """ Calculates the dc potential given the applied voltages and the stray field.
    Creates a third attribute of trap, called instance, a 3D matrix of potential values
    trap: file path and name, including '.pkl'
    VMULT: electrode voltages determined by the multipole control algorithm
    VMAN: electrode voltages determined by manual user control 
          e.g. VMAN  = [0,0,-2.,0,...] applies -2 V to electrode 3
    IMAN: array marking by an entry of 1 the electrodes which are under manual control, 
          e.g. IMAN = [0,0,1,0,...] means that electrode 3 is under manual control
    BOTH above conditions are necessary to manually apply the -2 V to electrode 3
    Ex,Ey,Ez: stray electric field; 3D matrices
    update: name to save new file as; typically the same as the name used for get_trap
    Nikos, cleaned up June 2013
    William, converted to Python Jan 2014"""
    import pickle
    #from all_functions import plot_potential
    file = open(trap,'rb')
    tf = pickle.load(file)
    file.close()
    p=tf.potentials # shorthand to refer to all potentials
    ne=tf.configuration.numElectrodes
    X,Y,Z=tf.potentials.X,tf.potentials.Y,tf.potentials.Z # grid vectors
    x,y,z=np.meshgrid(X,Y,Z)   
    [Ex,Ey,Ez]=E
    Vout = np.zeros((p['EL_DC_1'].shape[0],p['EL_DC_1'].shape[1],p['EL_DC_1'].shape[2]))
    # build up the potential from the normal DC elctrodes
    for ii in range(ne):
        Vout = Vout + VMULT[ii]*p['EL_DC_{}'.format(ii)]
    Vout = Vout-Ex*x-Ey*y-Ez*z
    # update the trapping field data structure with instance attributes
    tf.instance.DC=Vout
    tf.instance.RF=p.EL_RF # not needed, but may be useful notation
    tf.instance.X=X
    tf.instance.Y=Y
    tf.instance.Z=Z
    tf.instance.E=E
    tf.instance.check=True
    if update==True:
        name=trap
        print('Saving '+name+' as a data structure...')
        output = open(name,'wb')
        pickle.dump(tf,output)
        output.close()
    return tf.instance.DC
 
def set_voltages():
    """Provides the DC voltages for all DC electrodes to be set to using the parameters and voltage controls from analyze_trap.
    Outputs an array of values to set each electrode and used as VMULT for dc_potential in post_process_trap.
    The Ui and Ei values control the weighting of each term of the multipole expansion.
    In most cases, multipoleControls will be True, as teh alternative involves more indirect Mathiew calculations.
    Nikos, July 2009, cleaned up October 2013
    William Python 2014""" 
    #0) set parameters
    from project_parameters import savePath,name,multipoleControls,reg,driveFrequency,ax,az,phi,coefs,manuals
    import pickle
    trap = savePath+name+'.pkl'
    file = open(trap,'rb')
    tf = pickle.load(file)
    file.close()
    V,X,Y,Z=tf.instance.DC,tf.instance.X,tf.instance.Y,tf.instance.Z
    tc=tf.configuration
    C = tc.multipoleControl
    el = []
    #1) check if trap_knobs has been run yet, creating multipoleControl and multipoleKernel
    if tc.trap_knobs != True:
        return 'WARNING: You must run trap_knobs first!'
    #2a) determine electrode voltages directly
    elif multipoleControls: # note plurality to contrast from attribute
        el = np.dot(C,coefs.T)     # these are the electrode voltages
    #2b) determine electrode volages indirectly
    else:
        charge = tc.charge
        mass = tc.mass
        V0 = mass*(2*np.pi*frequencyRF)**2/charge
        U2 = az*V0/8
        U1 = U2+ax*V0/4
        U3 = 2*U1*np.tan(2*np.pi*(phi+tc.thetaRF)/180)
        U1p= np.sqrt(U1**2+U3**2/2)
        U4 = U1p*tc.Qrf[4]/tc.Qrf[1]
        U5 = U1p*tc.Qrf[5]/tc.Qrf[1]
        inp = np.array([E[0], E[1], E[2], U1, U2, U3, U4, U5]).T
        mCf = tc.multipoleCoefficients[1:9,:]
        el = np.dot(mCf.T,inp) # these are the electrode voltages
        el = np.real(el)
    #3) regularize if set to do so
    reg = 0
    if reg: 
        C = el
        Lambda = np.linalg.lstsq(tc.multipoleKernel,C)
        Lambda=Lambda[0]
        el = el-(np.dot(tc.multipoleKernel,Lambda))
    #4) Apply manual electrode values 
    for index in range(len(el)):
        if manuals[index]:
            el[index] = manuals[index] 
    return el 

def pfit(Vrf,Vdc,X,Y,Z,Irf,Jrf,Krf):
    """Find the secular frequencies, tilt angle, and position of the dc 
    saddle point for given combined input parameters. 
    fx,fy,fz are the secular frequencies
    theta is the angle of rotation from the p2d transformation (rotation)
    Depth is the distance between the potential at the trapping position and at the escape point
    Xdc,Ydc,Zdc are the coordinates of the trapping position
    Xe,Ye,Ze are the coordinates of the escape position
    William Python February 2014."""
    #1) find dc potential
    #from all_functions import plot_potential,p2d,trap_depth,find_saddle,exact_saddle
    from project_parameters import charge,mass,driveAmplitude,Omega,debug,scale
    #2) find pseudopotential
    """Gebhard, Oct 2010:
    changed back to calculating field numerically in ppt2 instead directly
    with bemsolver. this is because the slow bemsolver (new version) does not output EX, EY, EZ."""
    [Ey,Ex,Ez] = np.gradient(Vrf,abs(X[1]-X[0])/scale,abs(Y[1]-Y[0])/scale,abs(Z[1]-Z[0])/scale) # fortran indexing
    Esq = Ex**2 + Ey**2 + Ez**2
    #3) plotting pseudopotential, etc; outdated?
    PseudoPhi = Esq*(charge**2)/(4*mass*Omega**2) 
    U = PseudoPhi+charge*Vdc # total trap potential
    if debug.pfit:
#         plot_potential(Vrf,X,Y,Z,'1D plots','Vrf','U_{rf} (eV)',[Irf,Jrf,Krf])
#         plot_potential(Ex,X,Y,Z,'1D plots','Ex','U_{ps} (eV)',[Irf,Jrf,Krf])
#         plot_potential(Ey,X,Y,Z,'1D plots','Ey','U_{ps} (eV)',[Irf,Jrf,Krf])
#         plot_potential(Ez,X,Y,Z,'1D plots','Ez','U_{ps} (eV)',[Irf,Jrf,Krf])    
#         plot_potential(Esq,X,Y,Z,'1D plots','E**2','U_{ps} (eV)',[Irf,Jrf,Krf])
        plot_potential(Vrf,X,Y,Z,'1D plots','Vrf','U_{rf} (eV)',[Irf,Jrf,Krf])
        plot_potential(PseudoPhi/charge,X,Y,Z,'1D plots','Pseudopotential','U_{ps} (eV)',[Irf,Jrf,Krf])
#         plot_potential(Vdc,X,Y,Z,'1D plots','DC Potential','U_{sec} (eV)',[Irf,Jrf,Krf])
        plot_potential(U/charge,X,Y,Z,'1D plots','Trap Potential','U_{sec} (eV)',[Irf,Jrf,Krf])
    #4) determine trap frequencies and tilt in radial directions
    Uxy = U[Irf-3:Irf+4,Jrf-3:Jrf+4,Krf]
    MU = np.max(Uxy) # normalization factor, will be undone when calculating frequencies
    Uxy = Uxy/MU 
    nx,ny,nz=X.shape[0],Y.shape[0],Z.shape[0]
    x,y,z = np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x[i,j,k] = X[i]
                y[i,j,k] = Y[j]
                z[i,j,k] = Z[k]
    dL = x[Irf+3,Jrf,Krf]-x[Irf,Jrf,Krf] # is this X? Originally x. Temporarily y so that dL not 0. Probably related to meshgrid or indexing.
    xr = (x[Irf-3:Irf+4,Jrf-3:Jrf+4,Krf]-x[Irf,Jrf,Krf])/dL 
    yr = (y[Irf-3:Irf+4,Jrf-3:Jrf+4,Krf]-y[Irf,Jrf,Krf])/dL
    [C1,C2,theta] = p2d(Uxy,xr,yr) 
    C1,C2,theta = C1[0],C2[0],theta[0]                     
    fx = (1e3/dL)*np.sqrt(abs(2*C1*MU/(mass)))/(2*np.pi)
    fy = (1e3/dL)*np.sqrt(abs(2*C2*MU/(mass)))/(2*np.pi)
    #5) trap frequency in axial direction
    Uz=U[Irf,Jrf,:] # old projection
    l1 = np.max([Krf-6,1])
    l2 = np.min([Krf+6,np.max(Z.shape)])
    p = np.polyfit((Z[l1:l2+1]-Z[Krf])/dL,Uz[l1:l2+1],6)
    fz = (1e3/dL)*np.sqrt(2*p[4]/mass)/(2*np.pi)
    [Depth,Xe,Ye,Ze] = trap_depth(U/charge,X,Y,Z,Irf,Jrf,Krf)  
    return [fx,fy,fz,theta,Depth,Xe,Ye,Ze] 

def exact_saddle(V,X,Y,Z,dim,Z0=None):
    """This version finds the approximate saddle point using pseudopotential,
    does a multipole expansion around it, and finds the exact saddle point by
    maximizing the quadrupole terms. Similar to interpolation.
    V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    dim is the dimensionality (2 or 3).
    Z0 is the coordinate where a saddle point will be sought if dim==2.
    Nikos Daniilidis 9/1/09.
    Had issues with function nesting and variable scope definitions in Octave.
    Revisited for Octave compatibility 5/25/13.
    Needs Octave >3.6, pakages general, miscellaneous, struct, optim. 
    William Python Jan 2014"""
    #from all_functions import find_saddle,sum_of_e_field
    if dim==3:
        [I,J,K]=find_saddle(V,X,Y,Z,3) # guess saddle point; Z0 not needed
        r0=[X[I],Y[J],Z[K]]
        if I<2 or I>V.shape[0]-2: 
            print('exact_saddle.py: Saddle point out of bounds in radial direction.')
            return r0
        if J<2 or J>V.shape[1]-2:
            print('exact_saddle.py: Saddle point out of bounds in vertical direction.')
            return r0
        if K<2 or K>V.shape[2]-2:
            print('exact_saddle.py: Saddle point out of bounds in axial direction.')
            return r0
        if V.shape[0]>100:
            Vn = V[I-2:I+3,J-2:J+3,K-2:K+3] # create smaller 5x5x5 grid around the saddle point to speed up optimization
            # note that this does not prevent the optimization function from trying values outside this
            Xn,Yn,Zn=X[I-2:I+3],Y[J-2:J+3],Z[K-2:K+3] # change grid vectors as well
        else:
            Vn,Xn,Yn,Zn = V,X,Y,Z
        #################################### Minimize
        r=spo.minimize(sum_of_e_field,r0,args=(Vn,Xn,Yn,Zn)) 
        r=r.x # unpack for desired values
        Xs,Ys,Zs=r[0],r[1],r[2] 
    #################################################################################################    
    if dim==2: 
        if len(V.shape)==3:
            K=0 # in case there is no saddle
            for i in range(len(Z)):
                if Z[i-1]<Z0 and Z[i]>=Z0:
                    K=i-1
        Vs = V.shape
        if K>=Vs[1]: # Matlab had Z, not V; also changed from == to >=
            return('The selected coordinate is at the end of range.')
        v1=V[:,:,K-1] # potential to left
        v2=V[:,:,K] # potential to right (actually right at estimate; K+1 to be actually to right)
        V2=v1+(v2-v1)*(Z0-Z[K-1])/(Z[K]-Z[K-1]) # averaged potential around given coordinate
        [I,J,K0]=find_saddle(V,X,Y,Z,2,Z0) 
        r0=X[I],Y[J]
        if (I<2 or I>V.shape[0]-2): 
            print('exact_saddle.py: Saddle point out of bounds in radial direction.\n')
            return r0
        if (J<2 or J>V.shape[1]-2):
            print('exact_saddle.py: Saddle point out of bounds in vertical direction.\n')
            return r0
        if V.shape[0]>100:
            Vn = V[I-2:I+3,J-2:J+3,K-2:K+3] # create smaller 5x5x5 grid around the saddle point to speed up optimization
            # note that this does not prevent the optimization function from trying values outside this
            Xn,Yn,Zn=X[I-2:I+3],Y[J-2:J+3],Z[K-2:K+3] # Matlab 4, not 2
        else:
            Vn,Xn,Yn,Zn = V,X,Y,Z
        ################################## Minimize
        r=spo.minimize(sum_of_e_field_2d,r0,args=(Z0,Vn,Xn,Yn,Zn)) 
        r=r.x # unpack for desired values
        Xs,Ys,Zs=r[0],r[1],Z0
    return [Xs,Ys,Zs]
 
def find_saddle(V,X,Y,Z,dim,Z0=None):
    """Returns the indices of the local extremum or saddle point of the scalar A as (Is,Js,Ks).
    V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    Z0 is the Z axis index (may be a decimal) on which the saddle point is evaluated, if dim==2. 
    3/15/14: Z0 is coord, not index; Ks is the index
    For dim==2, the values of A are linearly extrapolated from [Z0] and [Z0]+1
    to those corresponding to Z0 and Ks is such that z[Ks]<Z0, z[Ks+1]>=Z0."""
    debug=False # internal code only; typically False
    from project_parameters import scale
    if (dim==2 and Z0==None):
        return 'z0 needed for evaluation'
    if dim==3:
        if len(V.shape)!=3:
            return('Problem with find_saddle.m dimensionalities.')
        f=V/float(np.amax(V)) # Normalize field
        [Ex,Ey,Ez]=np.gradient(f,abs(X[1]-X[0])/scale,abs(Y[1]-Y[0])/scale,abs(Z[1]-Z[0])/scale) # grid spacing is automatically consistent thanks to BEM-solver
        E=np.sqrt(Ex**2+Ey**2+Ez**2) # magnitude of gradient (E field)
        m=E[1,1,1]
        origin=[1,1,1]
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                for k in range(E.shape[2]):
                    if E[i,j,k]<m:
                        m=E[i,j,k]
                        origin=[i,j,k]          
        if debug:
            print('DEBUGGING...')
            fig=plt.figure()
            e=np.reshape(E,(1,E.shape[0]*E.shape[1]*E.shape[2]))
            ind,e=np.argsort(e),np.sort(e)
            e=e[0]
            ind=ind[0] #Sort V by the same indexing.
            v=np.reshape(V,(1,V.shape[0]*V.shape[1]*V.shape[2]))
            v=v[0]
            plt.plot(e/float(np.amax(e)))
            def index_sort(v,e):
                """Takes in two lists of the same length and returns the first sorted by the indexing of i sorted."""
                es=np.sort(e)
                ix=np.argsort(e)
                vs=np.ones(len(v)) #Sorted by the sorting defined by f being sorted. 
                # If v==e, this returns es.
                for i in range(len(v)):
                    j=ix[i]
                    vs[i]=v[j]
                return vs
            v=index_sort(v,e) # Is it supposed to look like this?
            plt.plot(v/float(np.amax(v)))
            plt.title('Debugging: blue is sorted gradient, green is potential sorted by gradient')
            plt.show() #f is blue and smooth, v is green and fuzzy.
        if origin[0]==(1 or V.shape[0]):
            print('find_saddle: Saddle out of bounds in  x (i) direction.\n')
            return origin
        if origin[0]==(1 or V.shape[1]):
            print('find_saddle: Saddle out of bounds in  y (j) direction.\n')
            return origin
        if origin[0]==(1 or V.shape[2]): 
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
            if Ks>=Vs[1]: # Matlab had Z, not V; also changed from == to >=
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
        if m>1e-4: # This requires a grid with step size 0.01, not just 0.1.
            if debug:
                Is,Js=np.NaN,np.NaN
                print('Warning, there seems to be no saddle point.')
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
 
def mesh_slice(V,n,X,Y,Z): 
    """Plots successive slices of matrix V in the direction given by n.
    n=0[I],1[J],2[K]
    X,Y,X are vectors that define the grid in three dimensions
    William Python Jan 2014
    """
    from matplotlib import cm
    import mpl_toolkits.mplot3d.axes3d as p3
    import time
    order=np.array([(1,2,0),(2,0,1),(0,1,2)])
    q=np.transpose(V,(order[n])) # See projection for why we could also use take instead.
    if n==0: # Make a less cumbersome and more consistent version of this?
        i,j=X,Y
        i,j=np.array([i]),np.array([j]).T
        I,J=i,j
        for m in range(j.shape[0]-1): # -1 because we already have the first row as I.
            I=np.vstack((I,i))
        for m in range(i.shape[1]-1):
            J=np.hstack((J,j))
    if n==1:
        i,j=Y,Z
        i,j=np.array([i]),np.array([j]).T
        I,J=i,j
        for m in range(j.shape[0]-1): # -1 because we already have the first row as I.
            I=np.vstack((I,i))
        for m in range(i.shape[1]-1):
            J=np.hstack((J,j))
    if n==2:
        i,j=Z,X
        i,j=np.array([i]),np.array([j]).T
        I,J=i,j
        for m in range(j.shape[0]-1): # -1 because we already have the first row as I.
            I=np.vstack((I,i))
        for m in range(i.shape[1]-1):
            J=np.hstack((J,j))
    labels={
        0:('horizontal axial (mm)','height (mm)'),
        1:('horizontal radial (mm)','horizontal axial (mm)'),
        2:('height (mm)','horizontal radial (mm)')
        }   
    class animated(object): # 4D, plots f(x,y,z0) specific to mesh_slice.
        def __init__(self,I,J,q):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.I,self.J=I,J
            self.q=q[:,0,:]
            self.surf=self.ax.plot_surface(self.J,self.I,self.q,cmap=cm.coolwarm,antialiased=False)
        def drawNow(self,ii,q,n):
            self.surf.remove()
            self.slc=q[:,ii,:]
            self.surf=self.ax.plot_surface(self.J,self.I,self.slc,cmap=cm.coolwarm,antialiased=False)
            plt.ylabel(labels[n][1])
            plt.xlabel(labels[n][0])
            #plt.title(ii) #Optional: this moves down during animation.
            plt.draw() # redraw the canvas
            time.sleep(0.01)
            self.fig.show()
    anim=animated(I,J,q)
    for ii in range(q.shape[1]):
        if ii==q.shape[1]-1:
            plt.title('Animation complete.')
        anim.drawNow(ii,q,n)
    return plt.show()
 
def plot_potential(V,X,Y,Z,key='1D plots',tit=None,ylab=None,origin=None): 
    """V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    Makes 2D mesh plots and 1D plots of 3D matrix around an origin
    key: 0: no plots, 1: 2D plots, 2: 1D plots, 3: both
    tit is the title of the plots to be produced.
    ylab is the label on the y axis of the plot produced.
    William Python Jan 2014"""
    print 'running plot_potential...',key
    if origin==None:
        origin=find_saddle(V,X,Y,Z,3)
    if (key==0 or key=='no plots'):
        return 
    if (key==1 or key=='2D plots' or key==3 or key=='both'): # 2D Plots, animation
        mesh_slice(V,0,X,Y,Z) 
        mesh_slice(V,1,X,Y,Z) 
        mesh_slice(V,2,X,Y,Z) 
    if (key==2 or key=='1D plots' or key==3 or key=='both'): # 1D Plots, 3 subplots
        ########## Plot I ##########
        axis=X
        projection=V[:,origin[1],origin[2]]
        fig=plt.figure()
        plt.subplot(2,2,1)
        plt.plot(axis,projection) 
        plt.title(tit)
        plt.xlabel('x (mm)')
        plt.ylabel(ylab)
        ######### Plot J ##########
        axis=Y
        projection=V[origin[0],:,origin[2]]
        plt.subplot(2,2,2)
        #plt.plot(axis[8:21],projection[8:21])
        plt.plot(axis,projection)
        plt.xlabel('y (mm)')
        plt.ylabel(ylab)
        ######### Plot K ##########
        axis=Z
        projection=V[origin[0],origin[1],:]
        plt.subplot(2,2,3)
        plt.plot(axis,projection)
        plt.xlabel('z (mm)')
        plt.ylabel(ylab)
        plt.show()
    return tit+' displayed'   

def p2d(V,x,y): 
    """Fits a 2D polynomial to the data in V, a 2D array of potentials.
    x and y are 2d Coordinate matricies.
    Returns Af, Bf, and theta; the curvatures of the Xr axis, Yr axes, and angle between X and Xr.
    We are not sure if the angle is correct when x and y are not centered on zero."""
    def s(a,N):
        """Shortcut function to convert array x into a coluumn vector."""
        a=np.reshape(a,(1,N**2),order='F').T
        return a
    N=V.shape[1]
    con=np.ones((x.shape[0],x.shape[1])) # constant terms
    xx,yy,xy=x*x,y*y,x*y
    xxx,yyy,xxy,xyy=xx*x,yy*y,xx*y,x*yy
    xxxx,yyyy,xxxy,xxyy,xyyy=xx*xx,yy*yy,xxx*y,xx*yy,x*yyy
    V2=s(V,N)    
    lst=[yyyy,xxxy,xxyy,xyyy,xxx,yyy,xxy,xyy,xx,yy,xy,x,y,con]
    Q=s(xxxx,N)
    count = 0
    for elem in lst:
        elem=s(elem,N)
        count+=1
        Q=np.hstack((Q,elem))
    c=np.linalg.lstsq(Q,V2) 
    c=c[0]
    theta=-0.5*np.arctan(c[11]/(c[10]-c[9]))
    Af=0.5*(c[9]*(1+1./np.cos(2*theta))+c[10]*(1-1./np.cos(2*theta)))
    Bf=0.5*(c[9]*(1-1./np.cos(2*theta))+c[10]*(1+1./np.cos(2*theta)))
    theta=180.*theta/np.pi
    return (Af, Bf, theta)
 
def plotN(trap,title=None,convention=None): 
    """Mesh the values of the DC voltage corresponding to the N DC electrodes of a planar trap,
    in a geometrically "correct" way.
    trap is a vector of N elements.
    Possible to add in different conventions later.
    Nikos, July 2009
    William Python 2014"""
    from matplotlib import cm 
    import mpl_toolkits.mplot3d.axes3d as p3
    N=trap.shape[0]
    n=np.floor(N/2)
    A=np.zeros((10*n,20))
    for i in range(int(n)): # Left electrodes.
        A[10*(n-i-1)+1:10*(n-i),1:3]=trap[i+1]
    A[:,5:7]=trap[0] # Left Center
    A[:,9:11]=trap[N-1] # Central electrode.
    A[:,13:15]=trap[0] # Right Center
    for i in range(1,int(n+1)): # Right electrodes.
        A[10*(n-i)+1:10*(n+1-i),17:19]=trap[i+n-1]
    x = np.arange(A.shape[0])
    y = np.arange(A.shape[1])
    x,y = np.meshgrid(x,y)
    z = A[x,y]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.hot, linewidth=0)
    fig.colorbar(surf)
    plt.title(title)
    return plt.show()

def spher_harm_bas(Xc,Yc,Zc,X,Y,Z,Order):
    """Make the real spherical harmonic matrix in sequence of [Y00,Y-11,Y01,Y11,Y-22,Y-12,Y02,Y12,Y22...]
    In other words: Calculate the basis vectors of the sph. harm. expansion:  """
    import math as mt
    from scipy.special import lpmv
    # Construct variables from axes; no meshgrid as of 6/4/14; no potential as of 6/12/14
    nx,ny,nz=X.shape[0],Y.shape[0],Z.shape[0]
    x,y,z = np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x[i,j,k] = X[i]-Xc
                y[i,j,k] = Y[j]-Yc
                z[i,j,k] = Z[k]-Zc
    x,y,z=np.ravel(x,order='F'),np.ravel(y,order='F'),np.ravel(z,order='F') 
    r,rt=np.sqrt(x*x+y*y+z*z),np.sqrt(x*x+y*y)
    # Normalize with geometric mean, 3/15/14 (most recently); makes error go down about order of magnitude
    rsort=np.sort(r)
    rmin=rsort[1] # first element is 0 
    rmax=rsort[len(r)-1] 
    rnorm=np.sqrt(rmax*rmin)
    r=r/rnorm
    # Construct theta and phi
    theta,phi=np.zeros(len(r)),np.zeros(len(r))
    for i in range(len(z)): #Set theta and phi to be correct. 10/19/13
        theta[i] = mt.atan2(rt[i],z[i])
        phi[i] = mt.atan2(y[i],x[i])
    # Make the spherical harmonic matrix in sequence of [Y00,Y-11,Y01,Y11,Y-22,Y-12,Y02,Y12,Y22...]
    # In other words: Calculate the basis vectors of the sph. harm. expansion: 
    Yj = np.zeros((nx*ny*nz,(Order+1)**2))
    fp = np.sqrt(1/(4*np.pi))
    Yj[:,0] = fp*np.sqrt(2)
    mc = 1
    for n in range(1,Order+1):
        for m in range(n+1):
            #ymn = np.sqrt((2*n+1)/(4*np.pi))*r**n*lpmv(m,n,np.cos(theta))
            ymn = r**n*lpmv(m,n,np.cos(theta))
            ymn = fp*ymn*np.sqrt((2*n+1))#/(4*np.pi))
            if m==0:
                #Yj[:,mc] = ymn
                Yj[:,mc+n] = ymn
                # better ordering means we need to change external mapping, as well as trap.configuration at start of expand_field
            else: # Nm is conversion factor to spherical harmonics, exclusing the sqrt(2*n+1/4pi) portion so that there is no coefficient to the m=0
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

def spher_harm_exp(V,Yj,scale):
    """Solves for the coefficients Mj that best combine with the basis to produce the potential field V."""
    # Convert the 3D DC potential into 1D array.
    # Numerically invert, here the actual expansion takes place and we obtain the expansion coefficients M_{j}.
    W=np.ravel(V,order='F') # 1D array of all potential values
    W=np.array([W]).T # make into column array
    Mj=np.linalg.lstsq(Yj,W)
    Mj=Mj[0] # array of coefficients
    Order = np.sqrt(len(Mj))-1
    # rescale to original units
    i = 0
    Order = int(np.sqrt(len(Mj))-1)
    for n in range(1,Order+1):
        for m in range(2*n+1):
            i += 1
            Mj[i] = Mj[i]/(scale**n)
    return Mj
 
def spher_harm_cmp(C,Yj,scale,Order): 
    """http://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    This function computes the potential V from the spherical harmonic coefficients,
    which used to be: V=C1*Y00+C2*Y10+C3*Y11c+C4*Y11s+...
    There the Ynm are chosen to be real, and subscript c corresponds to cos(m*phi) dependence,
    while s is sin(m*phi).
    Now it is: V=C1*Y00+C2*Y1-1+C3*Y10+C4*Y11+...
    Nikos June 2009"""
    import math as mt 
    from scipy.special import sph_harm
    V=[]
    if C.shape[0]!=(Order+1)**2:
        while True:
            st=raw_input('spherharrmcmp.py warning:\nSize of coefficient vector not equal to Order**2. Proceed? (y/n)\n')
            if st=='n':
                return
            elif st=='y':
                break
    i = 0
    for n in range(1,Order+1):
        for m in range(2*n+1):
            i += 1
            C[i] = C[i]*(scale**n)
    W=np.dot(Yj,C)
    N = len(W)
    n = int(N**(1/3.0))+1
    V=W.reshape(n,n,n,order='F').copy()
    return np.real(V)

def spher_harm_qlt(V,C,Xc,Yc,Zc,Order,Xe,Ye,Ze,tit):
    """This function determines the "quality" of the expansion of potential V in spherical harmonics
    It usd to be: (V=C00*Y00+C10*Y10+C11c*Y11c+C11s*Y11s+... )
    there the Ynm are chosen to be real, and subscript c corresponds to
    cos(m*phi) dependence, while s is sin(m*phi). 
    Now it is: (V=C00*Y00+C1-1*Y1-1+C10*Y10+C11*Y11+... )
    The expansion is carried up to multipoles of order Order.
    The indices in V are V(i,j,k)<-> V(x,y,z).
    V is the expanded potential.
    C is the coefficient vector.
    Xc,Yc,Zc are the coordinates of the center of the multipoles.
    Order is the order of the expansion.
    Xe,Ye,Ze are the vectors that define the grid in three directions.
    tit is a string describing the input potential for plot purposes. ('RF', 'DC', etc.).
    If title=='noplots' no plots are made.
    The coefficients are in the order:[C00,C1-1,C10,C11].T
    These correspond to the multipoles in cartesian coordinares:
    [c z -x -y (z^2-x^2/2-y^2/2) -3zx -3yz 3x^2-3y^2 6xy]
     1 2  3  4       5             6    7     8       9
    Nikos January 2009
    William Python Jan 2014"""
    #from all_functions import spher_harm_bas,spher_harm_cmp
    s=V.shape
    nx,ny,nz=s[0],s[1],s[2]
    basis = spher_harm_bas(Xc,Yc,Zc,Xe,Ye,Ze,Order)
    Vfit = spher_harm_cmp(C,basis,Order) 
    # subtract lowest from each and then normalize
    Vblock = np.ones((nx,ny,nz))
    Vfit = Vfit-Vblock*np.amin(Vfit)
    Vfit = Vfit/float(np.amax(Vfit))
    V = V-Vblock*np.amin(V)
    V = V/float(np.amax(V))
    dV = np.subtract(V,Vfit) 
    e = np.reshape(dV,(1,nx*ny*nz))
    e=abs(e)
    f_0=np.amax(e)
    f_1=np.mean(e)
    f_2=np.median(e)
    f = np.array([f_0,f_1,f_2])
    if tit=='noplots':
        return f
    plt.plot(e[0])
    plt.title(tit)
    plt.show() 
    return f
 
def sum_of_e_field(r,V,X,Y,Z,exact_saddle=True):
    """V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    r: center position for the spherical harmonic expansion
    Finds the weight of high order multipole terms compared to the weight of
    second order multipole terms in matrix V, when the center of the multipoles
    is at x0,y0,z0.
    Used by exact_saddle for 3-d saddle search.
    Note that order of outputs for spher_harm_exp are changed, but 1 to 3 should still be E field."""
    from project_parameters import debug
    x0,y0,z0=r[0],r[1],r[2]
    #from all_functions import spher_harm_exp
    basis,rnorm = spher_harm_bas(x0,y0,z0,X,Y,Z,3)
    c=spher_harm_exp(V,basis,rnorm) #Update these variables by abstraction.
    if debug.soef:
        print('Checking saddle: ({0},{1},{2})'.format(x0,y0,z0))
    s=c**2
    f=sum(s[1:4])/sum(s[4:9])
    real_f=np.real(f[0])
    if debug.soef:
        print('Optimization: {}'.format(real_f))
    return real_f

def sum_of_e_field_2d(r,z0,V,X,Y,Z,exact_saddle=True):
    """V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    r: center position for the spherical harmonic expansion
    Finds the weight of high order multipole terms compared to the weight of
    second order multipole terms in matrix V, when the center of the multipoles
    is at x0,y0,z0.
    Used by exact_saddle for 3-d saddle search.
    Note that order of outputs for spher_harm_exp are changed, but 1 to 3 should still be E field."""
    from project_parameters import debug
    x0,y0=r[0],r[1]
    #from all_functions import spher_harm_exp
    basis,rnorm = spher_harm_bas(x0,y0,z0,X,Y,Z,4)
    c=spher_harm_exp(V,basis,rnorm) #Update these variables by abstraction.
    if debug.soef:
        print('Checking saddle: ({0},{1},{2})'.format(x0,y0,z0))
    s=c**2
    f=sum(s[1:4])/sum(s[4:9])
    real_f=np.real(f[0])
    if debug.soef:
        print('Optimization: {}'.format(real_f))
    return real_f

def nullspace(A, atol=1e-13, rtol=0):
    """#3) Helper function From: http://wiki.scipy.org/Cookbook/RankNullspace
    Compute an approximate basis for the nullspace of A.
    The algorithm used by this function is based on the singular value decomposition of `A`.
    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.
    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.
    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero."""
    from numpy.linalg import svd
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns
 
def trap_depth_old(V,X,Y,Z,Im,Jm,Km,debug=False): 
    """Find the trap depth for trap potential V.
    Returns D,x,y,z.
    The trapping position is the absolute minimum in the potential function.
    The escape position is the nearest local maximum to the trapping position.
    D is the trap depth. This is the distance between the trapping and escape position.
        It is calculated along the vertical (X) direction
    x,y,z are the coordinates of the escape position.
    V is a cubic matrix of potential values
    X,Y,Z are vectors defining the grid in X,Y,Z directions.
    Im,Jm,Km are the indices of the trap potential minimum (ion position)."""  
    from project_parameters import debug
    #from all_functions import sum_of_e_field
    def a(a,N):
        """Shortcut function to convert array x into a row vector.""" 
        a=np.ravel(a, order='F') # Same order
        return a
    def index_sort(y,x):
        """Takes in two lists of the same length and returns y sorted by the indexing of x sorted."""
        xs=np.sort(x)
        ix=np.argsort(x)
        ys=np.ones(len(y)) #Sorted by the sorting defined by f being sorted. 
        for i in range(len(y)):
            j=ix[i]
            ys[i]=y[j]
        return ys
    if len(V.shape)!=3:
        return('Problem with find_saddle.py dimensionalities.\n')
    N1,N2,N3=V.shape
    N=N1*N2*N3
    f=V
    [Ex,Ey,Ez]=np.gradient(f,abs(X[1]-X[0]),abs(Y[1]-Y[0]),abs(Z[1]-Z[0]))
    E=np.sqrt(Ex**2+Ey**2+Ez**2)
    fs,Es=a(f,N),a(E,N) # Convert 3D to 1D array
    fs,Es=np.real(fs),np.real(Es)
    # identify the escape position and height by checking each point
    minElectricField=max(fs) # initialize as maximum E field magnitude
    distance=0
    escapeHeight=1
    escapePosition=[0,0,0]
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                if [i,j,k]==[Im,Jm,Km]:
                    Vm=V[i,j,k]
                elif E[i,j,k]<minElectricField:
                    minElectricField=E[i,j,k]
                    escapeHeight=V[i,j,k]
                    escapePosition=[i,j,k]
                    distance=abs(Im+Jm+Km-i-j-k)    
    if debug.trap_depth: # plot sortings of potential and electric field to view escape position
        plt.plot(np.sort(fs)) 
        plt.title('sorted potential field')
        plt.show()
        plt.plot(np.sort(Es)) 
        plt.title('sorted electric field')
        plt.show()
        q1=index_sort(fs,Es) 
        plt.title('potential field sorted by sorted indexing of electric field')
        plt.plot(q1)
        plt.show()
        q2=index_sort(Es,fs) 
        plt.title('electric field sorted by sorted indexing of potential field')
        plt.plot(q2)
        plt.show()      
    check=1 
    if debug.trap_depth: 
        print minElectricField,escapeHeight,escapePosition,distance   
    if distance<check:
        print('trap_depth.py: Escape point too close to trap minimum. Improve grid resolution or extend grid.')
    if escapeHeight>0.2:
        print('trap_depth.py: Escape point parameter too high. Improve grid resolution or extend grid.')
    D=escapeHeight-Vm
    [Ie,Je,Ke]=escapePosition
    [Xe,Ye,Ze]=[X[Ie],Y[Je],Z[Ke]]            
    return [D,Xe,Ye,Ze]

def trap_depth(V,X,Y,Z,Im,Jm,Km): 
    """Find the trap depth for trap potential V.
    Returns D,x,y,z.
    The trapping position is the absolute minimum in the potential function.
    The escape position is the nearest local maximum to the trapping position.
    D is the trap depth. This is the distance between the trapping and escape position.
        It is calculated along the vertical (X) direction
    x,y,z are the coordinates of the escape position.
    V is a cubic matrix of potential values
    X,Y,Z are vectors defining the grid in X,Y,Z directions.
    Im,Jm,Km are the indices of the trap potential minimum (ion position)."""  
    from project_parameters import debug,scale,position
    #from all_functions import sum_of_e_field,spher_harm_bas,spher_harm_exp,spher_harm_cmp,find_saddle
    def a(a,N):
        """Shortcut function to convert array x into a row vector.""" 
        a=np.ravel(a, order='F') # Same order
        return a
    N1,N2,N3=V.shape
    N=N1*N2*N3
    [Ex,Ey,Ez]=np.gradient(V,abs(X[1]-X[0])/scale,abs(Y[1]-Y[0])/scale,abs(Z[1]-Z[0])/scale)
    E=np.sqrt(Ex**2+Ey**2+Ez**2)
    # identify the escape position and height by checking each point
    minElectricField=np.max(E) # initialize as maximum E field magnitude
    distance=0
    escapeHeight=1
    escapePosition=[0,0,0]
    [Im,Jm,Km] = find_saddle(V,X,Y,Z,3)
    Vm = V[Im,Jm,Km]
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                if E[i,j,k]<minElectricField:
                    distance=abs(np.sqrt((Im-i)**2+(Jm-j)**2+(Km-k)**2)) 
                    if distance > 6:
                        minElectricField=E[i,j,k]
                        escapeHeight=V[i,j,k]
                        escapePosition=[i,j,k]
                        if debug.trap_depth: 
                            print E[i,j,k],V[i,j,k],[i,j,k],distance
    check=1 
    if debug.trap_depth: 
        print minElectricField,escapeHeight,escapePosition,distance   
    if distance<check:
        print('trap_depth.py: Escape point too close to trap minimum. Improve grid resolution or extend grid.')
    if escapeHeight>0.2:
        print('trap_depth.py: Escape point parameter too high. Improve grid resolution or extend grid.')
    D=escapeHeight-Vm
    [Ie,Je,Ke]=escapePosition
    [Xe,Ye,Ze]=[X[Ie],Y[Je],Z[Ke]]           
    return [D,Xe,Ye,Ze]

"""This script contains the parameters for developing the project trapping field.
Lower order functions sould rely entirely on data passed to them as configuration attributes."""

import numpy as np
import datetime
from treedict import TreeDict

# International System of Units
qe=1.60217646e-19 # charge
me=9.10938188e-31 # electron mass
mp=1.67262158e-27 # proton mass

# Universal Parameters
fourth = 0 # for testing synthetic data
save  = 1               
debug = TreeDict()
debug.import_data = 0
debug.get_trap = 0
debug.expand_field = 0
debug.trap_knobs = 0
debug.post_process_trap = 0
debug.pfit = 0
debug.soef = 0
debug.trap_depth = 0

#################################################################################
################################ import_data ####################################
#################################################################################
"""Includes project parameters relevant to import_data to build entire project in one script."""
simulationDirectory='C:\\Python27\\trap_simulation_software\\data\\text\\' # location of the text files
baseDataName = 'G_trap_field_12232013_wr40_' # Excludes the number at the end to refer to a set of text file simulations
projectName = 'abridged' # arbitrarily named by user
useDate = 0 # determine if simulation files are saved with our without date in name  
timeNow = datetime.datetime.now().date() # the present date and time 
fileName = projectName+'_'+str(timeNow)  # optional addition to name to create data structures with otherwise same name
if not useDate:
    fileName = projectName
simCount = [6,1]            # index of initial simulation and number of simulations; old nStart and nMatTot
dataPointsPerAxis = 21      # old NUM_AXIS 5, the number of data points along each axis of the cubic electrode potential
numElectrodes = 22          # old NUM_ELECTRODES, later nonGroundElectrodes, includes the first DC that is really RF
savePath = 'C:\\Python27\\trap_simulation_software\\data\\' # directory to save data at
scale = 1000. # based on BEM-solver grid units; we want mm internally, so if BEM is in microns, put 1000. (decimal for 2.7) here and grid vectors will be rescaled
perm = [0,2,1] 
###COORDINATES Nikos code uses y- height, z - axial, x - radial
#if drawing uses x - axial, y - radial, z - height, use perm = [1,2,0] (Euro trap)
#if drawing uses y - axial, x - radial, z - height, use perm = [0,2,1] (Sqip D trap, GG trap)
#if drawing uses Nikos or William (testconvention, use perm = [0,1,2]

#################################################################################
################################# get_trap ######################################
#################################################################################
"""fieldConfig, previously trapConfiguration; not all variables will be passed to output
Parameters used for get_trapping_field, expand_field, and trap_knobs
Some of the required parameters are listed with import config."""
position = 55/scale # trapping position along the trap axis (microns)
zMin = 5/scale      # lowest value along the rectangular axis
zMax = 105/scale    # highest value along the rectangular axis
zStep = 100/scale   # range of each simulation
r0 = 1              # scaling value, nearly always one
name = 'numerical8' # name of final, composite, single-simulation data structure; may also be string of choice              
trap = savePath+name+'.pkl'

#################################################################################
############################### expand_field ####################################
#################################################################################
Xcorrection = 0 # known offset from the RF saddle point
Ycorrection = 0 # known offset from the RF saddle point
regenOrder  = 9 # order to regenerate the data to, typically 2
E = [0,0,0]     # known electric field to correct for 
pureMultipoles = 0

#################################################################################
############################### trap_knobs ######################################
#################################################################################
trapFile = savePath+name+'.pkl'  
expansionOrder = 4 # order of multipole expansion, nearly always 2
assert expansionOrder <= regenOrder
reg = 0 # by regularization I mean minimizing the norm of el with addition of vectors belonging to the kernel of tf.config.multipoleCoefficients
""" Here we define the electrode combinations. 
The convention is physical electrode -> functional electrode.
If electrodes 1 and 2 are combined into one electrode, then enter [[1,1],[2,1],[3,2] ...]
If electrodes 1 and 4 are not in use (grounded), then enter [[1,0],[2,1],[3,2],[4,0] ...]
numElectrodes = nonGroundElectrodes (i.e. last) is the center electrode.
There are no longer RF electrodes included.
electrodeMapping determines the pairing. 
manualElectrodes determines the electrodes which are under manual voltage control. 
It has numElectrodes elements (i.e. they are not connected to an arbitrary voltage, not to multipole knobs).
All entries != 0 are under manual control, and entries = 0 are not under manual control.""" 
electrodeMapping = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],
                    [11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],[18,18],[19,19],[20,20],[21,21]]) # worry about this later
electrodes = np.zeros(numElectrodes) # 0 is RF, the rest are DC, and the final is the center; unselected are manual
multipoles = np.zeros((expansionOrder+1)**2) # 0 is constant, 1-3 are z (2nd), 4-8 are z**2 (6th), 9 to 15 are z**3, 16 to 25 are z**4 (20th)
electrodes[:] = 1
electrodes[0] = 0
multipoles[1:4] = 1
multipoles[6] = 1
multipoles[20] = 1

#################################################################################
##############################  post_process  ##################################
#################################################################################
# We no longer use findEfield or anything other than justAnalyzeTrap
# findCompensation = 0 # this will alwys be False
# findEfield       = 0 # this will always be False
justAnalyzeTrap  = 1 # do not optimize, just analyze the trap, assuming everything is ok
rfplot = '1D plots'  # dimensions to plot RF with plotpot, may be 'no plots', '1D plots', '2D plots', or 'both
dcplot = '1D plots'  # dimensions to plot DC with plotpot

# set_voltages, old trap operation parameters
weightElectrodes  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # VMAN for dcpotential_instance in expand_field
charge = qe
mass = 40*mp          # mass of the ion, 40 because of Ca has 20 protons and 20 neutrons
driveAmplitude = 50 # Applied RF amplitude for analysis, typically 100 mV 
driveFrequency = 50e6 # RF frequency for ppt3 analysis, typically 40 MHz
Omega = 2*np.pi*driveFrequency 

# multipole coefficients (only set this to 0 when doing 2nd order multipoles
multipoleControls = 1          # sets the control parameters to be the U's (true), or the alpha parameters

# valid if 0
ax = -2e-3  # Mathieu alpha_x parameter  
az = 4.5e-3 # Mathieu alpha_z parameter 
phi = 0     # Angle of rotation of DC multipole wrt RF multipole 

# valid if 1
coefs = np.zeros((expansionOrder+1)**2) # this is the array of desired weights to multipole coefficients
# Simply define it by indices. The first three (0,1,2) are the electric field terms (-y,z,-x). The 0th may be changed to the constant.
# Note that these are the opposite sign of the actual electric field, which is the negative gradient of the potential.
# The (x**2-y**2)/2 RF-like term has index 7 and the z**2 term has index 5.
# The z**3 term is index 11 and the z**4 term is index 19.
coefs[6] = 10*np.sqrt(4*np.pi/5)
coefs[20] = 0

if fourth:
    baseDataName = 'fourth'#'G_trap_field_12232013_wr40_' # Excludes the number at the end to refer to a set of text file simulations
    projectName = 'fourth' # arbitrarily named by user\
    fileName = projectName+'_'+str(timeNow)  # optional addition to name to create data structures with otherwise same name
    simCount = [1,1]            # index of initial simulation and number of simulations; old nStart and nMatTot
    dataPointsPerAxis = 11      # old NUM_AXIS 5, the number of data points along each axis of the cubic electrode potential
    numElectrodes = 8           # old NUM_ELECTRODES, later nonGroundElectrodes, includes the first DC that is really RF
    perm = [0,1,2] 
    scale = 1
    position = 0 # trapping position along the trap axis (microns)
    zMin = -5    # lowest value along the rectangular axis
    zMax = 6     # highest value along the rectangular axis
    zStep = 11   # range of each simulation
    name = 'fourth1' # name of final, composite, single-simulation data structure; may also be string of choice    
    trapFile = savePath+name+'.pkl'   
#     manualElectrodes = [1,1,0,0,0,0,1,0]#,0,0,0,0,0,0,0,0,0,0,0,0,0,0]   
#     electrodeMapping = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]])
    electrodes = np.zeros(numElectrodes) # 0 is RF, the rest are DC, and the final is the center; unselected are manual
    multipoles = np.zeros((expansionOrder+1)**2) # 0 is constant, 1-3 are z (2nd), 4-8 are z**2 (6th), 9 to 15 are z**3, 16 to 25 are z**4 (20th)
    #electrodes[2:numElectrodes] = 1
    #electrodes[6] = 0
    electrodes[:] = 1
    multipoles[1:4] = 1
    multipoles[6] = 1
    multipoles[20] = 1
    coefs[6] = 1#10*np.sqrt(4*np.pi/5)
    coefs[20] = 1
    
    

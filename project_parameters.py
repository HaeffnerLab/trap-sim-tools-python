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
SQIP = 0   # for testing with G_Trap
CCT = 0    # for testing with CCT trap
fourth = 0 # for testing synthetic data
save  = 1  # saves data to python pickle
debug = TreeDict()
debug.import_data = 0  # displays potential of every electrode for every simulation
debug.get_trap = 0     # displays the newly connected electrode potentials, unless there was only one simulation
debug.expand_field = 1 # displays the first 3 orders of multipole coefficient values
debug.trap_knobs = 0   # displays plots of multipole controls
debug.post_process_trap = 0 # displays plots of electrode values, RF potential, and DC potential
debug.pfit = 0         # displays plots of pseudopotential and trap potential
debug.soef = 0         # displays progress in exact_saddle optimizations
debug.trap_depth = 0   # displays assorted values for the final trap depth

#################################################################################
################################ import_data ####################################
#################################################################################
"""Includes project parameters relevant to import_data to build entire project in one script."""
#simulationDirectory='C:\\Python27\\trap_simulation_software\\data\\text\\' # location of the text files
#baseDataName = 'G_trap_field_12232013_wr40_' # Excludes the number at the end to refer to a set of text file simulations
simulationDirectory = '/home/dylan/trap_simulations/old-lattice/'
baseDataName = 'lattice_3d_trap'
projectName = 'lattice_3d_trap' # arbitrarily named by user
useDate = 0 # determine if simulation files are saved with our without date in name  
timeNow = datetime.datetime.now().date() # the present date and time 
fileName = projectName+'_'+str(timeNow)  # optional addition to name to create data structures with otherwise same name
if not useDate:
    fileName = projectName
simCount = [1,1]            # index of initial simulation and number of simulations; old nStart and nMatTot
dataPointsPerAxis = 101      # old NUM_AXIS 5, the number of data points along each axis of the cubic electrode potential
numElectrodes = 2          # old NUM_ELECTRODES, later nonGroundElectrodes, includes the first DC that is really RF
#savePath = 'C:\\Python27\\trap_simulation_software\\data\\' # directory to save data at
savePath = '/home/dylan/trap_simulations/old-lattice/'
scale = 1000. # based on BEM-solver grid units; we want mm internally, so if BEM is in microns, put 1000. (decimal for 2.7) here and grid vectors will be rescaled
perm = [0,1,2] 
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
position = 0/scale # trapping position along the trap axis (microns)
zMin = -600/scale      # lowest value along the rectangular axis
zMax = 600/scale    # highest value along the rectangular axis
zStep = 1200/scale   # range of each simulation
r0 = 1              # scaling value, nearly always one
name = 'old-lattice' # name of final, composite, single-simulation data structure; may also be string of choice              
trap = savePath+name+'.pkl'

#################################################################################
############################### expand_field ####################################
#################################################################################
Xcorrection = 0 # known offset from the RF saddle point
Ycorrection = 0 # known offset from the RF saddle point
regenOrder  = 2 # order to regenerate the data to, typically 2
E = [0,0,0]     # known electric field to correct for 
pureMultipoles = 0

#################################################################################
############################### trap_knobs ######################################
#################################################################################
trapFile = savePath+name+'.pkl'  
expansionOrder = 2 # order of multipole expansion, nearly always 2
assert expansionOrder <= regenOrder
reg = 0 # by regularization we mean minimizing the norm of el with addition of vectors belonging to the kernel of tf.config.multipoleCoefficients
"""Define the electrode and multipole mappings here. 
We want to know which rows and columns of the multipole coefficients we want to use for the inversion to teh multipole controls.
Each is initially an array of 0 with length equal to the number of electrodes or multipoles, respectively.
elMap - each indexed electrode is removed and added to the electrode indexed by each value
electrodes - each set to 0 will not be used for the inversion in trap_knobs; 0 is RF, the rest are DC, and the final is the center
manuals - each nonzero element turns off the index electrode and forces its voltage to be the specified value; units are in Volts
multipoles - each set to 0 will not be used for the inversion in trap_knobs; 0 is constant, 1-3 are z (2nd), 4-8 are z**2 (6th), 9 to 15 are z**3, etc."""
elMap = np.arange(numElectrodes) # default electrode mapping
#elMap[2] = 3 # clears electrode 2 and adds it to 3
electrodes = np.zeros(numElectrodes) # 0 is RF, the rest are DC, and the final is the center
multipoles = np.zeros((expansionOrder+1)**2) # 0 is constant, 1-3 are z (2nd), 4-8 are z**2 (6th), 9 to 15 are z**3, 16 to 25 are z**4 (20th)
electrodes[:] = 1 # turns on all electrodes
electrodes[0] = 0 # not made redundant by rfbias in manual because we may want RF turned off without setting it to anything
manuals = np.zeros(numElectrodes) # will typically use this in place of deactivating electrodes directly
manuals[0] = 0 # refer to first index for RFbias, setting to zero does nothing beyond setting electrodes[0]
# manuals[6] = 2 # sets the 6th DC electrode to be 2V
multipoles[0:9] = 1 # turns on all orders 0 to 2 multipoles
# multipoles[6] = 1 # turns on the DC multipole
# multipoles[16:25] = 1 # turns on all 4th order multipoles
# multipoles[8] = 1 # turns on eth RF multipole
# multipoles[:] = 1 # turns on all multipoles up to expansionOrder
for el in range(numElectrodes):
    if manuals[el]:
        electrodes[el] = 0      

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
# The (x**2-y**2)/2 RF-like term has index 8 and the z**2 term has index 6.
# The z**3 term is index 12 and the z**4 term is index 20.
# coefs[4] = 10
coefs[6] = 10 # default value to z**2 term, which varies from about 5 to 15
# coefs[8] = -65 # default value to RF term, which varies from about 0 to -65
# multipoles[20] = 1
# coefs[20] = -200 # default value to z**4 term, which varies from about 0 to -300
# coefs[4:9] /= np.sqrt(4*np.pi/5) # conversion factor for 2nd order
# coefs[16:25] /= np.sqrt(8*np.pi/3) # conversion factor for 4th order

# Debugging parameters for sytnthetic data.
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
    electrodes = np.zeros(numElectrodes) # 0 is RF, the rest are DC, and the final is the center; unselected are manual
    multipoles = np.zeros((expansionOrder+1)**2) # 0 is constant, 1-3 are z (2nd), 4-8 are z**2 (6th), 9 to 15 are z**3, 16 to 25 are z**4 (20th)
    #electrodes[2:numElectrodes] = 1
    #electrodes[6] = 0
    electrodes[:] = 1
    multipoles[1:4] = 1
    multipoles[6] = 1
    multipoles[20] = 1
    coefs[20] = 100
    
if CCT:
    baseDataName = 'A_fingers_e3_field-pt' # Excludes the number at the end to refer to a set of text file simulations
    projectName = 'atrap_cct' # arbitrarily named by user
    useDate = 0 # determine if simulation files are saved with our without date in name  
    timeNow = datetime.datetime.now().date() # the present date and time 
    fileName = projectName+'_'+str(timeNow)  # optional addition to name to create data structures with otherwise same name
    if not useDate:
        fileName = projectName
    simCount = [1,1]            # index of initial simulation and number of simulations; old nStart and nMatTot
    numElectrodes = 24          # old NUM_ELECTRODES, later nonGroundElectrodes, includes the first DC that is really RF
    position = 760/scale # trapping position along the trap axis (microns)
    zMin = 755/scale      # lowest value along the rectangular axis
    zMax = 765/scale    # highest value along the rectangular axis
    zStep = 10/scale   # range of each simulation
    name = 'CCT_y_test' # name of final, composite, single-simulation data structure; may also be string of choice              
    trap = savePath+name+'.pkl'
    trapFile = savePath+name+'.pkl' 
    elMap = np.arange(numElectrodes) # default electrode mapping
    #elMap[2] = 3 # clears electrode 2 and adds it to 3
    electrodes = np.zeros(numElectrodes) # 0 is RF, the rest are DC, and the final is the center
    multipoles = np.zeros((expansionOrder+1)**2) # 0 is constant, 1-3 are z (2nd), 4-8 are z**2 (6th), 9 to 15 are z**3, 16 to 25 are z**4 (20th)
    electrodes[:] = 1 # turns on all electrodes
    electrodes[0] = 0 # not made redundant by rfbias in manual because we may want RF turned off without setting it to anything
    manuals = np.zeros(numElectrodes) # will typically use this in place of deactivating electrodes directly
    manuals[0] = 0 # refer to first index for RFbias, setting to zero does nothing beyond setting electrodes[0]
    # manuals[6] = 2 # sets the 6th DC electrode to be 2V
    multipoles[0:9] = 1 # turns on all orders 0 to 2 multipoles
    for el in range(numElectrodes):
        if manuals[el]:
            electrodes[el] = 0      
    # valid if 1
    coefs = np.zeros((expansionOrder+1)**2) # this is the array of desired weights to multipole coefficients
    coefs[4] = 10 # default value to z**2 term, which varies from about 5 to 15
    coefs[6] = 10
    coefs[8] = 10
    driveAmplitude = 100 # Applied RF amplitude for analysis, typically 100 mV 
    driveFrequency = 35e6 # RF frequency for ppt3 analysis, typically 35 MH

if SQIP:
    baseDataName = 'G_trap_field_12232013_wr40_' # Excludes the number at the end to refer to a set of text file simulations
    projectName = 'SQIP_testing' # arbitrarily named by user
    useDate = 0 # determine if simulation files are saved with our without date in name  
    timeNow = datetime.datetime.now().date() # the present date and time 
    fileName = projectName+'_'+str(timeNow)  # optional addition to name to create data structures with otherwise same name
    if not useDate:
        fileName = projectName
    simCount = [6,1]            # index of initial simulation and number of simulations; old nStart and nMatTot
    numElectrodes = 21          # old NUM_ELECTRODES, later nonGroundElectrodes, includes the first DC that is really RF
    position = 55/scale # trapping position along the trap axis (microns)
    zMin = 5/scale      # lowest value along the rectangular axis
    zMax = 105/scale    # highest value along the rectangular axis
    zStep = 100/scale   # range of each simulation
    name = 'SQIP_testing' # name of final, composite, single-simulation data structure; may also be string of choice              
    trap = savePath+name+'.pkl'
    trapFile = savePath+name+'.pkl' 
    elMap = np.arange(numElectrodes) # default electrode mapping
    electrodes = np.zeros(numElectrodes) # 0 is RF, the rest are DC, and the final is the center
    multipoles = np.zeros((expansionOrder+1)**2) # 0 is constant, 1-3 are z (2nd), 4-8 are z**2 (6th), 9 to 15 are z**3, 16 to 25 are z**4 (20th)
    electrodes[:] = 1 # turns on all electrodes
    electrodes[0] = 0 # not made redundant by rfbias in manual because we may want RF turned off without setting it to anything
    manuals = np.zeros(numElectrodes) # will typically use this in place of deactivating electrodes directly
    manuals[0] = 0 # refer to first index for RFbias, setting to zero does nothing beyond setting electrodes[0]
    multipoles[0:9] = 1 # turns on all orders 0 to 2 multipoles
    for el in range(numElectrodes):
        if manuals[el]:
            electrodes[el] = 0    
    coefs = np.zeros((expansionOrder+1)**2) # this is the array of desired weights to multipole coefficients
    coefs[4] = 0 # default value to z**2 term, which varies from about 5 to 15
    coefs[6] = 10
    coefs[8] = -65
    driveAmplitude = 50 # Applied RF amplitude for analysis, typically 50 mV 
    driveFrequency = 50e6 # RF frequency for ppt3 analysis, typically 50 MH

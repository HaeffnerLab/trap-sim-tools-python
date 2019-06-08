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
save  = 1  # saves data to python pickle
debug = TreeDict()
debug.import_data = 0  # displays potential of every electrode for every simulation
debug.expand_field = 0 # displays the first 3 orders of multipole coefficient values
debug.trap_knobs = 1   # displays plots of multipole controls
debug.post_process_trap = 1 # displays plots of electrode values, RF potential, and DC potential
debug.pfit = 1         # displays plots of pseudopotential and trap potential
debug.soef = 1         # displays progress in exact_saddle optimizations
debug.trap_depth = 1   # displays assorted values for the final trap depth
trapType = 'HOA'

#################################################################################
################################ import_data ####################################
#################################################################################
"""Includes project parameters relevant to import_data to build entire project in one script."""
#simulationDirectory='C:\\Python27\\trap_simulation_software\\data\\text\\' # location of the text files
#baseDataName = 'G_trap_field_12232013_wr40_' # Excludes the number at the end to refer to a set of text file simulations
simulationDirectory = 'HOA_trap_v1/'
baseDataName = 'CENTRALonly'
dataPointsPerAxis = [941,13,15]      # number of data points along each axis of the potential
numElectrodes = 12          # includes RF
#savePath = 'C:\\Python27\\trap_simulation_software\\data\\' # directory to save data at
savePath = 'HOA_trap_v1/'
scale = 1
perm = [1,2,0] 
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
position = 0/scale # trapping position along the trap axis (mm)
zMin = -2.3085/scale      # lowest value along the rectangular axis
zMax = 2.3085/scale    # highest value along the rectangular axis
zStep = 0.005/scale   # range of each simulation
r0 = 1              # scaling value, nearly always one
name = 'HOA_DAC_CENTRAL' # name of final, composite, single-simulation data structure; may also be string of choice              
trapOut = savePath+name+'.pkl'

#################################################################################
############################### expand_field ####################################
#################################################################################
Xcorrection = 0 # known offset from the RF saddle point
Ycorrection = 0 # known offset from the RF saddle point
regenOrder  = 2 # order to regenerate the data to, typically 2
E = [0,0,0]     # known electric field to correct for 

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

multipoles[0:9] = 1 # turns on all orders 0 to 2 multipoles
   

#################################################################################
##############################  post_process  ##################################
#################################################################################
# We no longer use findEfield or anything other than justAnalyzeTrap
# findCompensation = 0 # this will alwys be False
# findEfield       = 0 # this will always be False
justAnalyzeTrap  = 0 # do not optimize, just analyze the trap, assuming everything is ok
rfplot = '2D plots'  # dimensions to plot RF with plotpot, may be 'no plots', '1D plots', '2D plots', or 'both
dcplot = '2D plots'  # dimensions to plot DC with plotpot

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

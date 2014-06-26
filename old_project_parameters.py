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
save  = 1               
debug = TreeDict()
debug.import_data = 0
debug.get_trap = 1
debug.expand_field = 0
debug.trap_knobs = 0
debug.post_process_trap = 1
debug.pfit = 1
debug.soef = 1
debug.trap_depth = 1

#################################################################################
################################ import_data ####################################
#################################################################################
"""Includes project parameters relevant to import_data to build entire project in one script."""
simulationDirectory='C:\\Python27\\trap_simulation_software\\data\\text\\' # location of the text files
baseDataName = 'synthetic-pt' # Excludes the number at the end to refer to a set of text file simulations
projectName = 'pictures' # arbitrarily named by user
useDate = 1 # determine if simulation files are saved with our without date in name  
timeNow = datetime.datetime.now().date() # the present date and time 
fileName = projectName+'_'+str(timeNow)  # optional addition to name to create data structures with otherwise same name
if not useDate:
    fileName = projectName
simCount = [1,2]            # index of initial simulation and number of simulations; old nStart and nMatTot
dataPointsPerAxis = 5       # old NUM_AXIS 5, the number of data points along each axis of the cubic electrode potential
numElectrodes = 14          # old NUM_ELECTRODES, later nonGroundElectrodes, includes the final DC that is really RF
savePath = 'C:\\Python27\\trap_simulation_software\\data\\' # directory to save data at
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
position = 0 # trapping position along the trap axis (microns)
zMin = -4    # lowest value along the rectangular axis
zMax = 4     # highest value along the rectangular axis
zStep = 5    # range of each simulation
name = projectName # name of final, composite, single-simulation data structure; may also be string of choice              

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
expansionOrder = 2 # order of multipole expansion, nearly always 2
reg            = 1 # select if using regularization
# (by regularization I mean minimizing the norm of el with addition of vectors belonging to the kernel of tf.config.multipoleCoefficients)
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
electrodeMapping = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],
                    [11,11],[12,12],[13,13]])             
manualElectrodes = [0,0,0,0,0,0,0,0,0,0,0,0,0] # IMAN for dcpotential_instance in expand_field
usedMultipoles   = [1,1,1,1,1,1,1,1]
# check to make sure mapping and manual are consistent with each other and the number of electrodes
last_map = 0
last_man = 2
assert electrodeMapping[-1][0] == numElectrodes-1 # excludes RF electrode
assert len(manualElectrodes) == numElectrodes-1
for elem in range(electrodeMapping.shape[1]-1):
    el = electrodeMapping[elem][0]
    map = electrodeMapping[elem][1]
    man = manualElectrodes[elem]
    if map == last_map:
        if man != last_man:
            raise Exception('project_parameters: electrode mapping does not match manual electrodes')
    last_map = map
    last_man = man
cut_map = electrodeMapping[-1][0]-electrodeMapping[-1][1] # number of electrodes that map redundantly
cut_man = np.sum(manualElectrodes) # number of manual electrodes, an unspecified number of which map redundantly
cut_both = cut_man - cut_map # number of manual electrodes which also map redunadantly
cut_true = cut_map + cut_both # number of electrodes mapped redundantly minus number manual ones not mapped redundantly
nue = electrodeMapping[-1][0] - cut_true
numUsedElectrodes = nue # old NUM_USED_ELECTRODES, numElectrodes minus the number overwritten by electrodeMapping or manual

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
weightElectrodes  = [0,0,0,0,0,0,0,0,0,0,0,0,0] # VMAN for dcpotential_instance in expand_field
assert len(weightElectrodes) == numElectrodes-1
for el in weightElectrodes:
    if weightElectrodes[el] == 0:
        assert manualElectrodes[el] == 0
    elif manualElectrodes[el] == 0:
        assert weightElectrodes[el] == 0
mass = 40*mp          # mass of teh ion, 40 because of Ca has 20 protons and 20 neutrons
driveAmplitude = 10e2 # Applied RF amplitude for analysis, typically 100 mV 
driveFrequency = 40e6 # RF frequency for ppt3 analysis, typically 40 MHz

# multipole coefficients
multipoleControls = 1          # sets the control parameters to be the U's (true), or the alpha parameters

# valid if 0
ax = -2e-3  # Mathieu alpha_x parameter  
az = 4.5e-3 # Mathieu alpha_z parameter 
phi = 0     # Angle of rotation of DC multipole wrt RF multipole 

# valid if 1
Ex,Ey,Ez = -0.001, -0.001, -0.001 # electric field terms, V/mm*2
U1,U2,U3,U4,U5 = 1,1,1,1,1     # DC Quadrupoles that you want the trap to generate at the ion, V/mm
# order: 0.5*(x**2-y**2), 0.5*(2*z**2-x**2-y**2), x*y, x*z, y*z
Ex,Ey,Ez = Ex*10**3,Ey*10**3,Ez*10**3                         # rescaling to mm as 1/r
U1,U2,U3,U4,U5 = U1*10**6,U2*10**6,U3*10**6,U4*10**6,U5*10**6 # rescaling to mm as 1/r^2

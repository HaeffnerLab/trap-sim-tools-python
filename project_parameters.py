"""This script contains the parameters for developing the project trapping field.
Lower order functions sould rely entirely on data passed to them as configuration attributes."""

import numpy as np
import scipy.io as io
import datetime
from treedict import TreeDict

# International System of Units
qe=1.60217646e-19
me=9.10938188e-31
mp=1.67262158e-27

# Universal Parameters
save  = 1       
useDate = 1 # determine if simulation files are saved with our without date in name          
debug = TreeDict()
debug.import_data = 0
debug.get_trap = 0
debug.expand_field = 0
debug.trap_knobs = 1
debug.post_process_trap = 1
debug.pfit = 1
debug.soef = 1
debug.trap_depth = 1
eurotrap = 1
rfplot = '1D plots' # dimensions to plot RF with plotpot
dcplot = '1D plots' # dimensions to plot DC with plotpot


#################################################################################
################################ import_data ####################################
#################################################################################
"""Includes project parameters relevant to import_data to build entire project in one script."""
simulationDirectory='C:\\Python27\\trap_simulation_software\\data\\text\\' # location of the text files
baseDataName = 'synthetic-pt' # Excludes the number at the end to refer to a set of text file simulations
projectName = 'pictures' # arbitrarily named by user
timeNow = datetime.datetime.now().date() # the present date and time 
fileName = projectName+'_'+str(timeNow)  # optional addition to name to create data structures with otherwise same name
if not useDate:
    fileName = projectName
simCount = [1,2]            # index of initial simulation and number of simulations; old nStart and nMatTot
dataPointsPerAxis = 5       # old NUM_AXIS 5, the number of sata points along each axis of the cubic electrode potential
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
Xcorrection = 0
Ycorrection = 0
regenOrder = 2
E = [0,0,0] # Old Ex,Ey,Ez    

#################################################################################
############################### trap_knobs ######################################
#################################################################################
expansionOrder = 2
reg=True
""" Here we define the electrode combinations. 
The convention is physical electrode -> functional electrode.
If electrodes 1 and 2 are combined into one electrode, then enter [1 1; 2 1; 3 2;...
If electrodes 1 and 4 are not in use (grounded), then enter [1 0; 2 1; 3 2; 4 0...
numElectrodes = nonGroundElectrodes (i.e. last) is the RF electrode.
nonGroundElectrodes-1 (i.e. before the RF) is the center electrode.
electrodeMapping determines the pairing. 
manualElectrodes determines the electrodes which are under manual voltage control. 
It has numElectrodes elements (i.e. they are not connected to an arbitrary voltage, not to multipole knobs).
All entries != 0 are under manual control, and entries = 0 are not under manual control."""  
electrodeMapping = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],
                    [11,11],[12,12],[13,13]])                                               
dcVoltages       = [1,1,1,1,1,1,1,1,1,1,1,1,1] # VMULT for expand_field before setdc can be run                
manualElectrodes = [0,0,0,0,0,0,0,0,0,0,0,0,0] # VMAN for dcpotential_instance in expand_field
weightElectrodes = [0,0,0,0,0,0,0,0,0,0,0,0,0] # IMAN for dcpotential_instance in expand_field
usedMultipoles   = [1,1,1,1,1,1,1,1]
# check to make sure mapping and manual are consistent with each other and the number of electrodes
last_map = 0
last_man = 2
assert electrodeMapping[-1][0] == numElectrodes-1 # excludes RF electrode
assert len(dcVoltages) == numElectrodes-1
assert len(manualElectrodes) == numElectrodes-1
assert len(weightElectrodes) == numElectrodes-1
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
################################# From ppt2  ####################################
#################################################################################
# set_voltages
multipoleControls = True          # sets the control parameters to be the U's (true), or the alpha parameters
# (by regularization I mean minimizing the norm of el with addition of vectors belonging to the kernel of tf.config.multipoleCoefficients)
mass = 40*mp # 40 because of Ca ions
driveAmplitude = 100     # Applied RF amplitude for ppt3 analysis
driveFrequency = 40e6    # RF frequency for ppt3 analysis
E = [.001,.001,.001]     # Stray electric field at the ion E = [Ex,Ey.Ez] (valid if multipoleControls == True); also in project_parameters
U1,U2,U3,U4,U5 = 1,1,1,1,1#.2,1,.2,0,0 # DC Quadrupoles that I want the trap to generate at the ion (valid if multipoleControls == True)
#U1,U2,U3,U4,U5 = 0.5*U1*10**-6,0.5*U2*10**-6,U3*10**-6,U4*10**-6,U5*10**-6 # rescaling to mm as 1/r^2
# also note the 0.5 for U1, U2
U1,U2,U3,U4,U5 = U3,U4,U2,U5,U1   # convert from convenient matlab order (U1=x**2-y**2) to mathematical order (U1=xy)
U2,U4 = U4,U2
# x*y,z*y,2*z**2-x**2-y**2,z*x,x**2-y**2 (python/math) to x**2-y**2,2*z**2-x**2-y**2,x*y,z*y,x*z (matlab/experiment)
# reweight the coefficients to match the spherical harmonic expansion
U1,U2,U3,U4,U5 = U1/5.6,U2/(-5.6),U3/27.45,U4/(-5.6),U5/11.2
Ex,Ey,Ez = Ex/4.25,Ez/(-6.02),Ey/4.25
ax = -0.002                       # Mathieu alpha_x parameter (valid only if multipoleControls == False)  
az = 4.5e-3                       # Mathieu alpha_z parameter (valid only if multipoleControls == False) 
phi = 0                           # Angle of rotation of DC multipole wrt RF multipole (valid only if multipoleControls == False)
# We no longer use findEfield or anoything other than justAnalyzeTrap
findEfield     = False   # determine the stray electric field for given dc voltages
justAnalyzeTrap  = True  # do not optimize, just analyze the trap, assuming everything is ok

if eurotrap:
    simulationDirectory='C:\\Python27\\trap_simulation_software\\data\\text\\' # location of the text files
    baseDataName = 'eurotrap-pt' # Excludes the number at the end to refer to a set of text file simulations
    projectName = 'eurotrap' # arbitrarily named by user
    timeNow = datetime.datetime.now().date() # the present date and time 
    fileName = projectName+'_'+str(timeNow)  # optional addition to name to create data structures with otherwise same name
    useDate = 1 # determine if simulation files are saved with our without date in name
    simCount = [1,6]            # index of initial simulation and number of simulations; old nStart and nMatTot
    dataPointsPerAxis = 21      # old NUM_AXIS 5, the number of sata points along each axis of the cubic electrode potential
    numElectrodes = 22          # old NUM_ELECTRODES, later nonGroundElectrodes, includes the final DC that is really RF
    perm = [1,2,0] 
    position = -535 # trapping position along the trap axis (microns)
    zMin = -630    # lowest value along the rectangular axis
    zMax = -510     # highest value along the rectangular axis
    zStep = 20    # range of each simulation
    name = 'eurotrap' # name of final, composite, single-simulation data structure 
    electrodeMapping = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],
                        [11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],[18,18],[19,19],[20,20],[21,21]])                                               
    dcVoltages       = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # VMULT for expand_field before setdc can be run                
    manualElectrodes = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # VMAN for dcpotential_instance in expand_field
    weightElectrodes = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # IMAN for dcpotential_instance in expand_field
    assert electrodeMapping[-1][0] == numElectrodes-1 # excludes RF electrode
    assert len(dcVoltages) == numElectrodes-1
    assert len(manualElectrodes) == numElectrodes-1
    assert len(weightElectrodes) == numElectrodes-1
    cut_map = electrodeMapping[-1][0]-electrodeMapping[-1][1] # number of electrodes that map redundantly
    cut_man = np.sum(manualElectrodes) # number of manual electrodes, an unspecified number of which map redundantly
    cut_both = cut_man - cut_map # number of manual electrodes which also map redunadantly
    cut_true = cut_map + cut_both # number of electrodes mapped redundantly minus number manual ones not mapped redundantly
    nue = electrodeMapping[-1][0] - cut_true
    numUsedElectrodes = nue # old NUM_USED_ELECTRODES, numElectrodes minus the number overwritten by electrodeMapping or manual

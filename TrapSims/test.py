import numpy as np
from simulation import simulation
import matplotlib.pyplot as plt

### EXAMPLE 

## (1) set up variables
path = '../HOA_trap_v1/CENTRALonly.pkl'
na = [941,13,15] # number of points per axis
ne = 12 # number of electrodes
perm = [1,2,0] 
position = [0,0.07,0] # approximate trapping position
order = 2

charge = 1.6021764e-19
mass = 1.67262158e-27 * 40
RF_ampl = 100 
RF_freq = 50e6 #in Hz

#(2) initialize the simulation and import the data
s = simulation(charge,mass)
s.import_data(path,ne,na,perm)

#(3) expand the potential for each electrode to the given spherical harmonic order around the given position
s.expand_field(position,order)

#plot potentials
fig,ax = plt.subplots(2,1)
for n in range(len(s.electrode_positions)):
    ax[0].plot(s.Z,s.electrode_potentials[n][6][7],label = str(s.electrode_names[n]))
    ax[1].plot(s.Z,s.electrode_potentials_regenerated[n][6][7],label = str(s.electrode_names[n]))
ax[0].legend()
ax[1].legend()
plt.show()

#plot multipoles
fig,ax = plt.subplots(9,1)
for i in range(9):
	Y = s.multipoles[:,i]
	Y = Y.reshape(s.nx,s.ny,s.nz)
	ax[i].imshow(Y[:,7,:])
plt.title('z,x (axial,radial) slices of multipoles')
plt.show()

#plot multipole coefficients
s.plot_multipoleCoeffs()

#(4) remove U4,U5,U6
usedMultipoles = np.zeros((s.expansion_order+1)**2)
usedMultipoles[0:6] = np.ones(6)
s.set_used_multipoles(usedMultipoles)

#(5) invert the expansion from (3) to get the multipole control matrix
s.multipole_control(False)

print s.multipoleControl

#plot voltages for each multipole
for n in range(len(s.multipoleControl)):
    s.plot_trapV(s.multipoleControl[n],"Multipole " + str(n))

#(6) print out the cFile for use in labrad
s.print_cFile(16,'test3.cls') # file includes *all* multipoles that should be available for the user in the gui.


#test HOA solution
vs_solution1 = np.zeros(ne)
vs_solution1[1] = -0.91
vs_solution1[3] = -1
vs_solution1[10] = -1
vs_solution1[9] = -0.919
coeffs_solution1 = np.dot(s.multipole_expansions,vs_solution1)
print coeffs_solution1

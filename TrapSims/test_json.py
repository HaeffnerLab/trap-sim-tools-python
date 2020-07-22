import numpy as np
from simulation import simulation
import matplotlib.pyplot as plt

### EXAMPLE 

## (1) set up variables
path = 'C:/Users/cm467/Documents/Jupyter/Oxionics/electrode_geometry/trap_test.json'
na = [21,21,21] # number of points per axis
ne = 10 # number of electrodes
perm = [1,2,0] # move from [ax, rad, height] to [rad, height, axial]
position = [0,40.0,0] # approximate trapping position
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
    ax[0].plot(s.Z,s.electrode_potentials[n][10][10],label = str(s.electrode_names[n]))
    ax[1].plot(s.Z,s.electrode_potentials_regenerated[n][10][10],label = str(s.electrode_names[n]))
ax[0].legend()
ax[1].legend()
plt.show()

#plot multipoles
fig,ax = plt.subplots(9,1)
fig.suptitle('z,x (axial,radial) slices of multipoles')
for i in range(9):
	Y = s.multipoles[:,i]
	Y = Y.reshape(s.nx,s.ny,s.nz)
	ax[i].imshow(Y[:,7,:])
plt.show()

fig,ax = plt.subplots()
fig.suptitle('z,x (axial,radial) slices of multipoles')
i = 0
Y = s.multipoles[:,i]
Y = Y.reshape(s.nx,s.ny,s.nz)
# ax[i].imshow(Y[:,7,:])
ax[i].imshow(Y[:,7,:])
plt.show()

#plot multipole coefficients for som electrodes
v1 = np.zeros(ne)
vs = []
for i in [0,1,2]:
	v = v1.copy()
	v[i] = 1
	vs.append(v.copy())
print (vs)
s.plot_multipoleCoeffs(vs,[s.electrode_names[i] for i in [0,1,2]])

#(4) remove U4,U5,U6
usedMultipoles = np.zeros((s.expansion_order+1)**2)
usedMultipoles[0:6] = np.ones(6)
s.set_used_multipoles(usedMultipoles)

#(5) invert the expansion from (3) to get the multipole control matrix
s.multipole_control(True)

print (s.multipoleControl)

#plot voltages for each multipole
for n in range(len(s.multipoleControl)):
    s.plot_trapV(s.multipoleControl[n],"Multipole " + str(n))

#(6) print out the cFile for use in labrad
# s.print_cFile(16,'test3.cls') # file includes *all* multipoles that should be available for the user in the gui.

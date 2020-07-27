# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from simulation import simulation
import matplotlib.pyplot as plt

### EXAMPLE 

## (1) set up variables
path = 'C:/Users/Clemens/Documents/Jupyter/electrode_geometry/trap_test.json'
na = [21,21,21] # number of points per axis
ne = 10 # number of electrodes
perm = [1,2,0] # move from [ax, rad, height] to [rad, height, axial]
position = [0,0.04,0] # approximate trapping position in mm
order = 2

charge = 1.6021764e-19
mass = 1.67262158e-27 * 40
RF_ampl = 100 
RF_freq = 50e6 #in Hz

# %% scrolled=true
#(2) initialize the simulation and import the data
s = simulation(charge,mass)
s.import_data(path,ne,na,perm)

#(3) expand the potential for each electrode to the given spherical harmonic order around the given position
s.expand_field(position,order)

# %% scrolled=false
# plot potentials, as imported to check if they look the same

X,Y = np.meshgrid(s.Z, s.X)
pots = np.array(s.electrode_potentials)
Z = pots[7, :,10,:]
print (Z.shape)
fig, ax = plt.subplots(dpi = 100)
ax.contour(X,Y,Z)
plt.show()
print (np.amin(Z),np.amax(Z))
print (s.electrode_names)

# %%
# plot potentials and compare to regenerated potentials. The lines should be symmetric about the trapping position
fig,ax = plt.subplots(2,1, figsize = (3.4,4), dpi = 200)
for n in range(len(s.electrode_positions)):
    ax[0].plot(s.Z,s.electrode_potentials[n][10][10],label = str(s.electrode_names[n]))
    ax[1].plot(s.Z,s.electrode_potentials_regenerated[n][10][10],label = str(s.electrode_names[n]))
ax[0].legend(fontsize = 9, loc='center right', bbox_to_anchor=(1.6, 0.5))
ax[1].legend(fontsize = 9, loc='center right', bbox_to_anchor=(1.6, 0.5))
plt.show()

# %%
# plot multipoles
fig,ax = plt.subplots(3,3, dpi = 200, sharex = 'all', sharey = 'all')
fig.suptitle('z,x (axial,radial) slices of multipoles')
for i in range(9):
    Y = s.multipoles[:,i]
    Y = Y.reshape(s.nx,s.ny,s.nz)
    ax[np.divmod(i,3)].imshow(Y[:,10,:])
plt.show()

# %% scrolled=true
# plot multipole coefficients for some electrodes
v1 = np.zeros(ne)
vs = []
for i in [0,1,2,3]:
    v = v1.copy()
    v[i] = 1
    vs.append(v.copy())
print (vs)
s.plot_multipoleCoeffs(vs,[s.electrode_names[i] for i in [0,1,2,3]])

# %%
# (4) optional: remove some multipoles
# usedMultipoles = np.zeros((s.expansion_order+1)**2)
# usedMultipoles[0:6] = np.ones(6)
# s.set_used_multipoles(usedMultipoles)

#(5) invert the expansion from (3) to get the multipole control matrix
s.multipole_control(True)

# %% scrolled=false
# make some bar plots of the multipole control vectors
multipoleControl = np.array(s.multipoleControl)

# this plot shows the multipole coefficients for all electrodes. Some symmetries should be visible!
fig, axs = plt.subplots(2,np.int(ne/2), dpi = 200, sharex = 'all', sharey = 'all')
for i in np.arange(ne):
    axs[np.divmod(i,np.int(ne/2))].bar(np.arange(9),multipoleControl[:,i])
plt.tight_layout()
plt.show()

# plots of individual multipole coefficients for all electrodes
# fig, axs = plt.subplots(2,np.int(ne/2), dpi = 200, sharex = 'all', sharey = 'all')
# for i in np.arange(ne):
#     axs[np.divmod(i,np.int(ne/2))].bar(np.arange(1),multipoleControl[4:5,i])
# plt.tight_layout()
# plt.show()

# %% scrolled=false
#(6) print out the cFile for use in labrad
# s.print_cFile('Cfile_test.txt')# file includes *all* multipoles that should be available for the user in the gui.
#(7) optional: print out the same cFile in slightly different format and includes header. Not useful for DAC
s.print_cFile_general('Cfile_test_general.txt')

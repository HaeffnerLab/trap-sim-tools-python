# %%
import pickle
# add multipoles package path
import sys

from Electrodes.multipoles import MultipoleControl
from plottingfuncns import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper_functions import *

fin = "vtks/htrapF_mega_short0.01_size100.0_DC1_mesh"
strs = "DC1 DC2 DC3 DC4 DC5 DC6 DC7 DC8 DC9 DC10 DC11 DC12 DC13 DC14 DC15 DC16 DC17 DC18 DC19 DC20 DC21".split()
fout = "htrap_example"
pathgrid = './gridExample.pkl'

fgrid = open(pathgrid, 'rb')
grid = pickle.load(fgrid)

write_pickle(fin,fout,grid,strs)
# %% md
# import data, and define parameters
# %%
path = './htrap_simulation_1.pkl'


f = open(path, 'rb')
trap = pickle.load(f)

zl = 3.7*72*1e-3
xl = -0.051*72*1e-3
yl = 1.06*72*1e-3

position = [xl, yl, zl]
nROI = 5
roi = [nROI, nROI, nROI]
order = 2
trap['Z'] = np.array(trap['Z'])
trap['X'] = np.array(trap['X'])
trap['Y'] = np.array(trap['Y'])
controlled_electrodes = []
strs = "DC1 DC2 DC3 DC4 DC5 DC6 DC7 DC8 DC9 DC10 DC11 DC12 DC13 DC14 DC15 DC16 DC17 DC18 DC19 DC20 DC21".split()
excl = {
    "DC6": "gnd",
        "DC4": "gnd",
        "DC5": "gnd",
        "DC8": "gnd",
        "DC14": "DC13",
        "DC11": "gnd",
        "DC12": "gnd"
}
for electrode in strs:
    if electrode in excl and excl[electrode] != "gnd":
        trap['electrodes'][excl[electrode]]["potential"] = trap['electrodes'][excl[electrode]]["potential"] + \
                                                           trap['electrodes'][electrode]["potential"]
    elif electrode not in excl:
        controlled_electrodes.append(electrode)

used_order1multipoles = ['Ex', 'Ey', 'Ez']
used_order2multipoles = ['U1', 'U2', 'U3', 'U5']
used_multipoles = used_order1multipoles + used_order2multipoles
print(used_multipoles)
position = np.array(position)

x0_ind = (np.abs(trap['X'] - position[0])).argmin()
y0_ind = (np.abs(trap['Y'] - position[1])).argmin()
z0_ind = (np.abs(trap['Z'] - position[2])).argmin()
# %% md
# create object
# %%
print(trap['Y'])
# %%
s = MultipoleControl(trap, position, roi, controlled_electrodes, used_multipoles, order)
print('Multipole names:', s.multipole_names)
print('Normalization factors:', s.normalization_factors)
# %%
# s.multipole_expansion
# %%
# plot multipole coefficients vs multipole names for each electrode at a certain height
# %%
v1 = pd.Series(np.zeros(len(controlled_electrodes)), index=controlled_electrodes)
vs = []
for ele in s.electrode_names:
    v = v1.copy()
    v[ele] = 1
    vs.append(v)


# print vs




# plot_multipole_vs_expansion_height(5)
# %% md
# plot multipole coefficients vs different heights for each electrode
# %%
height_list = np.round(trap['Y'][nROI:] * 1e3)
numMUltipoles = len(s.multipole_print_names)
ne = len(s.electrode_names)
multipoles_vs_height = np.zeros((len(height_list), numMUltipoles, ne))
print(height_list)
for i, height in enumerate(height_list):
    position1 = [xl, height * 1e-3, zl]
    s.update_origin_roi(position1, roi)
    multipoles_vs_height[i] = np.asarray(s.multipole_expansion.loc[s.multipole_names])

size = 15
fig, ax = plt.subplots(numMUltipoles, 1, figsize=(20, 60))





for i, mul in enumerate(s.multipole_print_names):
    for j, ele in enumerate(s.electrode_names[0:1]):
        ax[i].plot(height_list, multipoles_vs_height[:, i, j], label=ele)
        ax[i].set_title(mul, fontsize=size)
        ax[i].set_xticks(height_list)
        #         ax[i].set_xlim(left=50, right=100)
        ax[i].tick_params(labelsize=size)
        ax[i].set_xlabel('Height (um)', fontsize=size)
    update_colors(ax[i])
    ax[i].legend(fontsize=size, bbox_to_anchor=(1, 1))

fig.canvas.draw()
fig.tight_layout(pad=1)




plot_muls(s,xl,zl,roi,height= 75, ez=0, ex=0, ey=0,u2=10, u5=0, u1=0, u3=0)
# %% md
# plot coefficients can be achieved for each multipole (controlled individually) when apply max 40 volts
# %%
height_list = np.round(trap['Y'][nROI:] * 1e3)
numMUltipoles = len(used_multipoles)
Coeffs = pd.DataFrame()
for height in height_list:
    position1 = [xl, height * 1e-3, zl]
    s.update_origin_roi(position1, roi)

    Coeffs_temp = pd.Series()
    for key in used_multipoles:
        multipole_coeffs = pd.Series(np.zeros(len(used_multipoles)), index=used_multipoles)
        multipole_coeffs[key] = 1
        voltages = s.setMultipoles(multipole_coeffs)
        max_v = np.max(abs(voltages))
        Coeffs_temp[key] = 40 / max_v

    Coeffs[height] = Coeffs_temp

size = 25
# plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 12))
for mul in used_order1multipoles:
    ax1.plot(height_list, Coeffs.loc[mul], label=s.multipole_print_names[mul])
ax1.set_ylabel(r'1st order multipoles $(V/mm)$', fontsize=size)
ax1.set_xticks(np.arange(height_list[0], height_list[-1] + 1, 5))
ax1.set_xlabel('Height (um)', fontsize=size)
ax1.tick_params(labelsize=size)
ax1.legend(fontsize=size)

ax1.set_ylim(0, 2)  # set ylim

ax1.grid(True)

for mul in used_order2multipoles:
    ax2.plot(height_list, Coeffs.loc[mul], label=s.multipole_print_names[mul])
ax2.set_ylabel(r'2nd order multipoles $(V/mm^2)$', fontsize=size)
ax2.set_xticks(np.arange(height_list[0], height_list[-1] + 1, 5))
ax2.set_xlabel('Height (um)', fontsize=size)

ax2.set_ylim(0, 30)  # set ylim




ax2.tick_params(labelsize=size)
ax2.legend(fontsize=size)
ax2.grid(True)
fig.suptitle('Multipole coefficients when apply 10V (Ion Trap 14, exclude U4)', fontsize=size)
fig.tight_layout(pad=1)

secax = ax2.secondary_yaxis('right', functions=(U2_to_mhz, mhz_to_U2))
secax.tick_params(labelsize=size, colors='#ff7f0e')
secax.set_ylabel('$Ca^{+}$ trap frequency (MHz)', fontsize=size, color='#ff7f0e')


# plt.savefig('Multipole_coeffs_20v_rfbias.jpg', format = 'jpg', dpi = 300)

plot_1d(s,xl,zl,roi,height=75, ez=0, ex=0, ey=0, u2=10,u5=0, u1=0, u3=0)
# %%
outarray = []
s.write_txt('el3_4-5-6-8-11-12-gnd_13-14', strs, excl)


plot_U2(s,xl,zl,roi,height=75, ez=0, ex=0, ey=0, u2=40, u5=0, u1=0, u3=0)
# output = interactive_plot.children[-1]
# output.layout.height = '1000px'
# %%
##################### BREAK POINT- none of the code below this runs properly for me yet ###################
# a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# np.ravel(a, 'F')
# # %%
# multipole_names = ['Ey', 'Ez', 'Ex', 'U3', 'U2', 'U5', 'U1']
# height_list = [50,60,70,80,90,100]
# volts_all_heights = []
# for height in height_list:
#     position1 = [xl, height * 1e-3 ,zl]
#     s.update_origin_roi(position1, roi)
#     volts_all_elecs = []
#     for multip in multipole_names:
#         multipole_coeffs = {'Ey': 0, 'Ez': 0, 'Ex': 0, 'U3': 0, 'U2': 0, 'U5': 0, 'U1': 0}
#         multipole_coeffs[multip] = 1
#         voltages = s.setMultipoles(multipole_coeffs)
#         volts_all_elecs.append(voltages.values)
#     volts_all_heights.append(np.ravel(volts_all_elecs, order='C'))
# volts_all_heights = np.transpose(volts_all_heights)
# # %%
# volts_all_heights.shape
# # %%
# import csv
#
# header1 = ['multipoles: Ey, Ez, Ex, U3, U2, U5, U1']
# header2 = ['default position: 100']
# with open('3d_trap14_cfile.csv', 'w', newline='') as csvfile:
#     cfile = csv.writer(csvfile, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
#     cfile.writerow(header1)
#     cfile.writerow(header2)
#     cfile.writerows(list(volts_all_heights))
#     cfile.writerow(height_list)
# # %%
# cfile_path = '3d_trap14_cfile.csv'
# cfile_text = open(cfile_path).read().split('\n')[:-1]
# # %%
# cfile_text[0].split('ultipoles:')[1].replace(' ', '').split(',')
# # %%
# cfile_text
# # %%
#
# # %%
# height_list1 = np.arange(80, 201, 1)
# m_ca = 6.66e-26
# omega_rf = 2 * np.pi * 52.05e6
# e = 1.6e-19
# v_per_mhz = []
# for height in height_list1:
#     position1 = [xl,height * 1e-3,zl]
#     s.update_origin_roi(position1, roi)
#     rf_quad = (s.multipole_expansion['RF2']['U3'] - s.multipole_expansion['RF1']['U3']) / 2 * 1e6
#     v_per_mhz.append(m_ca * omega_rf * 2 * np.pi / np.sqrt(2) / rf_quad / e * 1e6)
# # %%
# s.update_origin_roi([xl, yl, zl], roi)
# # s.multipole_expansion['RF1']['U3']
# # %%
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(height_list1, v_per_mhz)
# ax.set_ylim(6, 14)
# size = 16
# ax.tick_params(labelsize=size)
# ax.set_ylabel('voltage(v) / secular frequency (MHz)', fontsize=size)
# ax.set_xlabel('distance from substrate (um)', fontsize=size)
# # %%

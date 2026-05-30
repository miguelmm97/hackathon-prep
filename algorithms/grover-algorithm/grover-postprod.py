#%% Modules setup

# Math and plotting
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
import itertools

# Managing system, data and config files
from functions import load_my_data, load_my_attr

#%% Loading data
file_list = ['Experiment1.h5', 'Experiment2.h5', 'Experiment3.h5', 'Experiment4.h5', 'Experiment5.h5',  'Experiment6.h5',
             'Experiment7.h5', 'Experiment8.h5', 'Experiment9.h5', 'Experiment10.h5', 'Experiment11.h5', 'Experiment12.h5',
             'Experiment13.h5', 'Experiment14.h5', 'Experiment15.h5', 'Experiment16.h5']
data_dict = load_my_data(file_list, '../../data-grover')

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
palette1 = seaborn.color_palette(palette='Blues_r', n_colors=2)
palette2 = seaborn.color_palette(palette='Greens_r', n_colors=2)
palette3 = seaborn.color_palette(palette='PuRd_r', n_colors=2)
palette4 = seaborn.color_palette(palette='Purples_r', n_colors=2)
palette5 = seaborn.color_palette(palette='Reds_r', n_colors=2)
palette6 = seaborn.color_palette(palette='Wistia_r', n_colors=2)
palette7 = seaborn.color_palette(palette='winter', n_colors=2)
palette8 = seaborn.color_palette(palette='bone', n_colors=2)
markers = itertools.cycle((('^', '7'), ('o', '4')))
palettes = [(4, palette1), (5, palette2), (6, palette3), (7, palette4), (8, palette5), (9, palette6), (10, palette7),
            (13, palette8)]


# Axes
fig1 = plt.figure(figsize=(9, 6))
ax1 = fig1.gca()
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 0.3)
ax1.set_xlabel("$t/t_{opt}$", fontsize=20)
ax1.set_ylabel("$\\langle l \\rangle$", fontsize=20)
ax1.tick_params(which='major', width=0.75, labelsize=15)
ax1.tick_params(which='major', length=10, labelsize=15)
ax1.set_title('Expected value of the scale of information')

fig2 = plt.figure(figsize=(9, 6))
ax2 = fig2.gca()
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 0.17)
ax2.set_xlabel("$t/t_{opt}$", fontsize=20)
ax2.set_ylabel("$\\langle l \\rangle$", fontsize=20)
ax2.tick_params(which='major', width=0.75, labelsize=15)
ax2.tick_params(which='major', length=10, labelsize=15)
ax2.set_title('Median of the scale of information')

# Plot data
for i, key in enumerate(data_dict.keys()):
    marker, markersize = next(markers)

    # Parameters of the simulation
    num_qubits     = data_dict[key]['Parameters']['num_qubits']
    n_iter         = data_dict[key]['Parameters']['n_iter']
    optimal_iter   = data_dict[key]['Parameters']['opt_iter']
    marked_state   = data_dict[key]['Parameters']['marked_state_str']
    # marked_state   = data_dict[key]['Parameters']['marked_state']
    l_rescaled = np.arange(0, num_qubits) / (num_qubits - 1)
    t_rescaled = np.arange(0, n_iter + 1) / optimal_iter

    # Data of the simulation
    info_per_scale = data_dict[key]['Simulation']['info_per_scale']
    mean_info      = data_dict[key]['Simulation']['mean_info']
    median_info    = data_dict[key]['Simulation']['median_info']


    # Figure info
    for n_tuple in palettes:
        if num_qubits == n_tuple[0]: palette = n_tuple[1]

    # Expected value
    ax1.plot(t_rescaled, mean_info, marker=marker, markersize=markersize, color=palette[i % 2],
             label=f'$n=$ {num_qubits}, 'f'$|\psi_m\\rangle=$ {marked_state}')
    ax1.plot(t_rescaled, mean_info, color=palette[i % 2], alpha=0.1)

    # Median
    ax2.plot(t_rescaled, median_info, marker=marker, markersize=markersize, color=palette[i % 2],
             label=f'$n=$ {num_qubits}, 'f'$|\psi_m\\rangle=$ {marked_state}')
    ax2.plot(t_rescaled, median_info, color=palette[i % 2], alpha=0.1)

ax1.legend(loc='best', ncol=4, fontsize=7, frameon=False)
ax2.legend(loc='best', ncol=4, fontsize=7, frameon=False)
plt.show()
fig1.savefig('mean_scale.pdf', format='pdf', backend='pgf')
fig2.savefig('median_scale.pdf', format='pdf', backend='pgf')
#%% Imports

# Math and plotting
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn

# Managing data
import h5py
import os
import sys
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Information lattice and algorithm
from functions import (calc_info, plot_info_latt, calc_info_per_scale, grover_oracle, get_fileID, store_my_data,
                       attr_my_data, median)



#%% Definitions and parameters

# Circuit details
phi0_str         = '0000'
marked_states    = ['0001']
num_qubits       = len(phi0_str)

# Iteration number
dim_H = 2 ** num_qubits
theta0 = 2 * np.arccos(np.sqrt((dim_H - len(marked_states)) / dim_H))
optimal_iter1 = math.floor(math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2 ** num_qubits))))
optimal_iter2 = 0.5 * (np.pi / theta0 - 1)
n_iter = optimal_iter1
final_prob = np.sin((2 * n_iter + 1) * theta0 / 2)

# Definitions
info_dict = {}
info_per_scale = np.zeros((n_iter + 1, num_qubits))
mean_info = np.zeros((n_iter + 1,))
median_info = np.zeros((n_iter + 1, ))
l_rescaled = np.arange(0, num_qubits) / (num_qubits - 1)
t_rescaled = np.arange(0, n_iter + 1) / optimal_iter1
norm_info_per_scale = num_qubits
neg_tol = -1e-12

# Circuit instance
oracle = grover_oracle(marked_states)
grover_op = GroverOperator(oracle, insert_barriers=True)
qc = QuantumCircuit(num_qubits)
psi0 = Statevector.from_label(phi0_str)
if psi0.num_qubits != num_qubits:
    raise ValueError('Number of qubits of initial and marked states does not coincide.')


#%% Information lattice and information per scale

# Circuit evolution: Hadamard transform
qc.h(range(num_qubits))
info_dict[0] = calc_info(psi0.evolve(qc).data)
state = psi0.evolve(qc)
# Debug
for key in info_dict[0].keys():
    if any(info_dict[0][key] < neg_tol):
        raise ValueError(f'Negative mutual information: step {0} | scale {key}')


# Circuit evolution: Grover iterations and information lattice
for i in range(1, n_iter + 1):
    print(f'iter: {i} / {n_iter}')
    qc.compose(grover_op, inplace=True)
    info_dict[i] = calc_info(state.evolve(grover_op).data)
    state = state.evolve(grover_op)
    # Debug
    for key in info_dict[i].keys():
        if any(info_dict[i][key] < neg_tol):
            raise ValueError(f'Negative mutual information: step {i} | scale {key}')


# Information per scale
for step in info_dict.keys():
    info_per_scale[step, :] = calc_info_per_scale(info_dict[step], bc='open')
    # Debug
    if not np.allclose(np.sum(info_per_scale[step, :]), num_qubits, atol=1e-15):
        raise ValueError(f'Information per scale does not add up! Error: {np.abs(num_qubits - np.sum(info_per_scale[step]))}')


# Statistical parameters
for step in range(info_per_scale.shape[0]):
    prob_scale = info_per_scale[step, :] / norm_info_per_scale
    mean_info[step] = np.dot(prob_scale, l_rescaled)
    median_info[step] = median(l_rescaled, prob_scale)
    # Debug
    if not np.allclose(np.sum(prob_scale), 1, atol=1e-15):
        raise ValueError(f'Probability distribution is not normalised')

# Expected information and success probability
expected_state = np.zeros((dim_H, ), dtype=np.complex128)
success_probability = 0
for psi in marked_states:
    expected_state[int(psi, 2)] = 1.
    success_probability += state.data[int(psi, 2)] ** 2
expected_state = expected_state / np.linalg.norm(expected_state)
expected_info = calc_info(expected_state)
# Debug
if np.allclose(success_probability, final_prob, atol=1e-10):
    print(f'Theoretical probability and success probability not matching, diff: {np.abs(success_probability - final_prob)}')



#%% Saving data

file_list = os.listdir('../../data-grover')
expID = get_fileID(file_list, common_name='Experiment')
filename = '{}{}{}'.format('Experiment', expID, '.h5')
filepath = os.path.join('../../data-grover', filename)

with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'info_per_scale',  info_per_scale)
    store_my_data(simulation, 'mean_info',       mean_info)
    store_my_data(simulation, 'median_info',     median_info)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'marked_states',    marked_states)
    store_my_data(parameters, 'num_qubits',       num_qubits)
    store_my_data(parameters, 'phi0_str',         phi0_str)
    store_my_data(parameters, 'n_iter',           n_iter)
    store_my_data(parameters, 'opt_iter',         optimal_iter1)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
palette1 = seaborn.color_palette(palette='Blues', n_colors=n_iter+1)
palette2 = seaborn.color_palette(palette='dark:salmon_r', n_colors=n_iter+1)

# Information lattice per iteration
fig1 = plt.figure(figsize=(8, 5))
gs = GridSpec(2, int(np.ceil(n_iter / 2) + 1), figure=fig1)
for i in range(n_iter + 1):
    if ((n_iter + 1) % 2) == 0:
        if i < int((n_iter + 1)/ 2):
            ax = fig1.add_subplot(gs[0, i])
        else:
            ax = fig1.add_subplot(gs[1, i % int((n_iter + 1)/ 2)])
    else:
        if i <= int((n_iter + 1) / 2):
            ax = fig1.add_subplot(gs[0, i])
        elif i != (n_iter + 1) - 1:
            ax = fig1.add_subplot(gs[1, (i % int((n_iter + 1) / 2)) - 1])
        else:
            ax = fig1.add_subplot(gs[1, int((n_iter + 1) / 2) - 1])
    plot_info_latt(info_dict[i], ax)
    ax.set_title(f't: {i / optimal_iter1:.2f}')
fig1.suptitle(f'Initial state: {phi0_str} , marked state: {marked_states},  optimal iteration: {optimal_iter1}')

# Information per scale per iteration
fig2 = plt.figure(figsize=(8, 6.5))
ax2 = fig2.gca()
for step in info_dict.keys():
    ax2.plot(l_rescaled, info_per_scale[step], marker="o", label=f't = {step / optimal_iter1:.2f}', color=palette1[step])
    ax2.plot(l_rescaled, info_per_scale[step], color=palette1[step])
ax2.set_xlim(0, 1)
ax2.set_ylim(1e-2, 10)
ax2.set_yscale('log')
ax2.set_xlabel("$l/l_{max}$", fontsize=20)
ax2.set_ylabel("$\log{(i_l)}$", fontsize=20)
ax2.tick_params(which='major', width=0.75, labelsize=15)
ax2.tick_params(which='major', length=10, labelsize=15)
ax2.legend(loc='best', ncol=2, fontsize=10, frameon=False)
ax2.set_title(f'Initial state: {phi0_str} , marked state: {marked_states},  optimal iteration: {optimal_iter1}')


fig3 = plt.figure(figsize=(6, 5))
gs = GridSpec(2, 6, figure=fig3)
ax3_1 = fig3.add_subplot(gs[:, :2])
ax3_2 = fig3.add_subplot(gs[:, 2:])

qc.draw(output="mpl", style="iqp", ax=ax3_1)
grover_op.decompose().draw(output="mpl", style="iqp", ax=ax3_2)
fig3.tight_layout()

fig4 = plt.figure(figsize=(6, 5))
ax4 = fig4.gca()
plot_info_latt(expected_info, ax4)






plt.show()
# fig1.savefig(f'{expID}-info-lattice.pdf', format='pdf', backend='pgf')
# fig2.savefig(f'{expID}-info-scale.pdf', format='pdf', backend='pgf')
fig3.savefig(f'grover.pdf', format='pdf', backend='pgf')
#fig4.savefig(f'{expID}-algorithm.pdf', format='pdf', backend='pgf', bbox_inches='tight')



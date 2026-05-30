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
from qiskit.circuit.library import HGate, SwapGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Algorithm and information lattice
from functions import qft_block, calc_info, plot_info_latt, calc_info_per_scale

#%% Variables
num_qubits = 4                                            # Number of qubits
state = Statevector.from_label('1011')                    # Initial state
n_iter = num_qubits - 1 + int(np.floor(num_qubits / 2))   # Number of information measurements
info_dict = {}
title_dict = {}

#%% Main
qc = QuantumCircuit(num_qubits)
qc_plot = QuantumCircuit(num_qubits)

for n in range(num_qubits - 1, -1, -1):

    # H and series of controlled phase gates for each qubit
    if n != 0:
        # Information lattice
        state = state.evolve(qft_block(n, num_qubits))
        info_dict[num_qubits - 1 - n] = calc_info(state.data)
        title_dict[num_qubits - 1 - n] = f'Block qubit {n}'
        # Plotting the circuit
        qc_plot.compose(qft_block(n, num_qubits), inplace=True)
        qc_plot.barrier(label=f'Block qubit {n}')

    # H on the last qubit and swap operations
    else:
        # Information lattice
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        state = state.evolve(qc)
        info_dict[num_qubits - 1] = calc_info(state.data)
        title_dict[num_qubits - 1] = f'Block qubit {n}'
        # Plotting the circuit
        qc_plot.h(0)
        qc_plot.barrier(label=f'Block qubit {n}')

        for i in range(num_qubits - 1, int(np.floor(num_qubits / 2)) - 1, -1):
            # Information lattice
            qc = QuantumCircuit(num_qubits)
            qc.swap(i, num_qubits - i - 1)
            state = state.evolve(qc)
            info_dict[2 * num_qubits - 1 - i] = calc_info(state.data)
            title_dict[2 * num_qubits - 1 - i] = f'SWAP {i} - {num_qubits - 1 - i}'
            # Plotting the circuit
            qc_plot.swap(i, num_qubits - i - 1)
            qc_plot.barrier(label=f'SWAP {i} - {num_qubits - 1 - i}')

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig1 = qc_plot.draw(output="mpl", style="iqp")
ax1 = fig1.gca()

fig2 = plt.figure(figsize=(8, 5))
gs = GridSpec(2, int(np.ceil(n_iter / 2) + 1), figure=fig2)
for i in range(n_iter + 1):
    if ((n_iter + 1) % 2) == 0:
        if i < int((n_iter + 1)/ 2):
            ax = fig2.add_subplot(gs[0, i])
        else:
            ax = fig2.add_subplot(gs[1, i % int((n_iter + 1) / 2)])
    else:
        if i <= int((n_iter + 1) / 2):
            ax = fig2.add_subplot(gs[0, i])
        elif i != (n_iter + 1) - 1:
            ax = fig2.add_subplot(gs[1, (i % int((n_iter + 1) / 2)) - 1])
        else:
            ax = fig2.add_subplot(gs[1, int((n_iter + 1) / 2) - 1])
    plot_info_latt(info_dict[i], ax)
    ax.set_title(title_dict[i])
plt.show()




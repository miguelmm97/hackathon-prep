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

# Algorithm
from functions import qft_circuit, Umod_multi,  calc_info, plot_info_latt, calc_info_per_scale

#%% Variables
# Variables
m = int(np.ceil(np.log2(15)))           # Qubits in the second register (phase estimation)
t = 3                                   # Qubits in the first register (QFT)
n_iter = t + 2                          # Number of information measurements
psi0 = '0' * (t + m)                    # Initial state
state = Statevector.from_label(psi0)    # Initial state
info_dict = {}
title_dict = {}

#%% Main: U(x) = 2x mod 15
U = QuantumCircuit(m)
U.swap(3, 2)
U.swap(2, 1)
U.swap(1, 0)
Umod = U.to_gate()
psi_try = Statevector.from_label('0010').evolve(U)


# Repeated application of the powers of the U operator
qc_shor = QuantumCircuit(t + m, t)
qc_shor.h(range(t))
qc_shor.x(t)
state = state.evolve(qc_shor)
info_dict[0] = calc_info(state.data)
title_dict[0] = 'H + X'
qc_shor.barrier(label='H + X')

for idx in range(t):
    # Information lattice
    state = state.evolve(Umod_multi(m, idx, Umod), [idx] + list(range(t, t + m)))
    info_dict[idx + 1] = calc_info(state.data)
    title_dict[idx + 1] = f'U({2 ** idx})'
    # Plotting the circuit
    qc_shor.append(Umod_multi(m, idx, Umod), [idx] + list(range(t, t + m)))
    qc_shor.barrier(label=f'U({2 ** idx})')

# Inverse Quantum Fourier transform
qft_inv = qft_circuit(t).inverse()
qft_inv.name = 'QFT$^\dagger$'
state = state.evolve(qft_inv)
info_dict[n_iter - 1] = calc_info(state.data)
title_dict[n_iter - 1] = 'QFT$^\dagger$'
qc_shor.append(qft_inv, range(t))
qc_shor.barrier(label='QFT$^\dagger$')



#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


fig1 = qc_shor.draw(output="mpl", style="iqp")

fig2 = U.draw(output="mpl", style="iqp")
fig2.savefig(f'2mod15gate.pdf', format='pdf', backend='pgf')

fig3 = plt.figure(figsize=(8, 5))
gs = GridSpec(1, n_iter, figure=fig3)
for i in range(n_iter):
    ax = fig3.add_subplot(gs[0, i])
    plot_info_latt(info_dict[i], ax)
    ax.set_title(title_dict[i])
fig3.savefig(f'2mod15-info.pdf', format='pdf', backend='pgf')
plt.show()





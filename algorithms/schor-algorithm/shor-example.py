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
from functions import qft_circuit, Umod_multi

#%% Variables
m = int(np.ceil(np.log2(15)))   # Qubits in the second register (phase estimation)
t = 4                       # Qubits in the first register (QFT)


#%% Main: example circuit for U(x) = 7x mod 15

# Create 7mod15 gate
Umod = QuantumCircuit(m)
Umod.x(range(m))
Umod.swap(1, 2)
Umod.swap(2, 3)
Umod.swap(0, 3)
Umod = Umod.to_gate()

# Repeated application of the powers of the U operator
qc_shor = QuantumCircuit(t + m, t)
qc_shor.h(range(t))
qc_shor.x(t)
for idx in range(t - 1):
    qc_shor.append(Umod_multi(m, idx, Umod), [idx] + list(range(t, t + m)))

# Inverse Quantum Fourier transform
qft_inv = qft_circuit(t).inverse()
qft_inv.name = 'QFT$^\dagger$'
qc_shor.append(qft_inv, range(t))

# Circuit
fig1 = qc_shor.draw(output="mpl", style="iqp")
fig1.savefig('shor.pdf', format='pdf', backend='pgf')
plt.show()


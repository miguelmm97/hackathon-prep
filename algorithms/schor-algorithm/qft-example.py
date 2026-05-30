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
from functions import qft_circuit

#%% Main: example circuit
num_qubits = 4
qc = qft_circuit(num_qubits)
fig1 = qc.draw(output="mpl", style="iqp")


fig1.savefig('qft.pdf', format='pdf', backend='pgf')
plt.show()


#%% Imports

# Math and plotting
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from numpy import pi

# Managing data
import h5py
import os
import sys
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, SwapGate, MCMT, XGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Algorithm and information lattice
from functions import Umod_multi, qft_circuit, qft_gate, calc_info, plot_info_latt, calc_info_per_scale

# Figure settings
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#%% Main: Controlled rotations in QFT

# Cphase with basis states
qc_cphase = QuantumCircuit(2)
qc_cphase.h(1)
qc_cphase.cp(pi / 2, 1, 0)
psi0 = Statevector.from_label('00')
psi = psi0.evolve(qc_cphase)
info_latt_cphase = calc_info(psi.data)


# Cphase with superposition states
qc_cphase2 = QuantumCircuit(2)
qc_cphase2.h(0)
qc_cphase2.h(1)
qc_cphase2.cp(pi / 2, 1, 0)
psi0 = Statevector.from_label('00')
psi = psi0.evolve(qc_cphase2)
info_latt_cphase2 = calc_info(psi.data)


# Figure
fig1 = plt.figure(figsize=(6, 3))
gs = GridSpec(2, 2, figure=fig1)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[1, 0])
ax4 = fig1.add_subplot(gs[1, 1])
qc_cphase.draw(output="mpl", style="iqp", ax=ax1)
plot_info_latt(info_latt_cphase, ax=ax2)
qc_cphase2.draw(output="mpl", style="iqp", ax=ax3)
plot_info_latt(info_latt_cphase2, ax=ax4)
fig1.suptitle('Information profile of c-phase gates used for the QFT')
fig1.savefig('cphase.pdf', format='pdf', backend='pgf')

#%% Main: SWAP gates

# Swap gate with unentangled qubits
qc_swap1 = QuantumCircuit(2)
qc_swap1.x(1)
qc_swap1.swap(0, 1)
psi0 = Statevector.from_label('00')
psi = psi0.evolve(qc_swap1)
info_latt_swap1 = calc_info(psi.data)


# Swap gate with entangled qubits
qc_swap2 = QuantumCircuit(2)
qc_swap2.x(1)
qc_swap2.h(0)
qc_swap2.compose(MCMT(XGate(), 1, 1), inplace=True)
psi0 = Statevector.from_label('00')
psi = psi0.evolve(qc_swap2)
info_latt_swap2_1 = calc_info(psi.data)

qc_swap2.barrier()
qc_swap2.swap(0, 1)
psi = psi0.evolve(qc_swap2)
info_latt_swap2_2 = calc_info(psi.data)
qc_swap2.barrier()


# Swap gate with many entangled qubits
qc_swap3 = QuantumCircuit(4)
qc_swap3.x(1)
qc_swap3.h(1)
qc_swap3.cx(1, 2)
qc_swap3.h(0)
qc_swap3.cx(0, 3)
psi0 = Statevector.from_label('0011')
psi = psi0.evolve(qc_swap3)
info_latt_swap3_1 = calc_info(psi.data)

qc_swap3.barrier()
qc_swap3.swap(0, 1)
psi = psi0.evolve(qc_swap3)
info_latt_swap3_2 = calc_info(psi.data)
qc_swap3.barrier()


# Figure
fig2 = plt.figure(figsize=(6, 6))
gs = GridSpec(3, 4, figure=fig2)
ax1 = fig2.add_subplot(gs[0, 0:2])
ax2 = fig2.add_subplot(gs[0, 2:])
ax3 = fig2.add_subplot(gs[1, 0:2])
ax4 = fig2.add_subplot(gs[1, 2])
ax5 = fig2.add_subplot(gs[1, 3])
ax6 = fig2.add_subplot(gs[2, 0:2])
ax7 = fig2.add_subplot(gs[2, 2])
ax8 = fig2.add_subplot(gs[2, 3])
qc_swap1.draw(output="mpl", style="iqp", ax=ax1)
plot_info_latt(info_latt_swap1, ax=ax2)
qc_swap2.draw(output="mpl", style="iqp", ax=ax3)
plot_info_latt(info_latt_swap2_1, ax=ax4)
plot_info_latt(info_latt_swap2_2, ax=ax5)
qc_swap3.draw(output="mpl", style="iqp", ax=ax6)
plot_info_latt(info_latt_swap3_1, ax=ax7)
plot_info_latt(info_latt_swap3_2, ax=ax8)
fig2.savefig('swap.pdf', format='pdf', backend='pgf')
fig2.suptitle('Information profile of SWAP gates')


#%% Main: U-gates applied in Shor's algorithm
m = int(np.ceil(np.log2(15)))   # Qubits in the second register (phase estimation)
t = 1                          # Qubits in the first register (QFT)
psi0 = '0' * (t + m)
psi0 = Statevector.from_label(psi0)

Umod = QuantumCircuit(m)
Umod.x(range(m))
Umod.swap(1, 2)
Umod.swap(2, 3)
Umod.swap(0, 3)
Umod = Umod.to_gate()

# Umod gate without superposition
qc_Umod = QuantumCircuit(m + t)
qc_Umod.x(t)
qc_Umod.append(Umod_multi(m, 0, Umod), [0] + list(range(t, t + m)))
psi = psi0.evolve(qc_Umod)
info_latt_Umod = calc_info(psi.data)

# Umod gate with superposition
qc_Umod2 = QuantumCircuit(m + t)
qc_Umod2.x(t)
qc_Umod2.h(range(t))
qc_Umod2.append(Umod_multi(m, 0, Umod), [0] + list(range(t, t + m)))
psi = psi0.evolve(qc_Umod2)
info_latt_Umod2 = calc_info(psi.data)

# Higher powers of Umod gate
qc_Umod3 = QuantumCircuit(m + t)
qc_Umod3.x(t)
qc_Umod3.h(range(t))
qc_Umod3.append(Umod_multi(m, 1, Umod), [0] + list(range(t, t + m)))
psi = psi0.evolve(qc_Umod3)
info_latt_Umod3 = calc_info(psi.data)


# Figure
fig3 = plt.figure(figsize=(5, 8))
gs = GridSpec(1, 4, figure=fig3)
ax1 = fig3.add_subplot(gs[0, 0])
ax2 = fig3.add_subplot(gs[0, 1])
ax3 = fig3.add_subplot(gs[0, 2])
ax4 = fig3.add_subplot(gs[0, 3])
# ax5 = fig3.add_subplot(gs[2, 0:2])
# ax6 = fig3.add_subplot(gs[2, 2:])
qc_Umod.draw(output="mpl", style="iqp", ax=ax1)
plot_info_latt(info_latt_Umod, ax=ax2)
qc_Umod2.draw(output="mpl", style="iqp", ax=ax3)
plot_info_latt(info_latt_Umod2, ax=ax4)
# qc_Umod3.draw(output="mpl", style="iqp", ax=ax5)
# plot_info_latt(info_latt_Umod3, ax=ax6)
fig3.savefig('U-info.pdf', format='pdf', backend='pgf')
fig3.suptitle('Information profile of c-U gates used in Shors algorithm')



#%% Main: QFT with different input states

# Full QFT with a particular superposition
qc_QFT = QuantumCircuit(4)
qc_QFT.h(range(3))
qft = qft_gate(4)
qft.name = 'QFT'
qc_QFT = qc_QFT.compose(qft, range(4))
psi0 = Statevector.from_label('0000')
psi = psi0.evolve(qc_QFT)
info_latt_QFT = calc_info(psi.data)

# Full QFT with a particular superposition
qc_QFT2 = QuantumCircuit(4)
qc_QFT2.h(range(2))
qft = qft_gate(4)
qft.name = 'QFT'
qc_QFT2 = qc_QFT2.compose(qft, range(4))
psi0 = Statevector.from_label('0000')
psi = psi0.evolve(qc_QFT2)
info_latt_QFT2 = calc_info(psi.data)

# Full QFT with entanglement
qc_QFT3 = QuantumCircuit(4)
qc_QFT3.h(1)
qc_QFT3.cx(1, 2)
psi0 = Statevector.from_label('0000')
psi = psi0.evolve(qc_QFT3)
info_latt_QFT3_1 = calc_info(psi.data)
qc_QFT3.barrier()

qft = qft_gate(4)
qft.name = 'QFT'
qc_QFT3 = qc_QFT3.compose(qft, range(4))
psi = psi0.evolve(qc_QFT3)
info_latt_QFT3_2 = calc_info(psi.data)
qc_QFT3.barrier()





# Figure
fig4 = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 4, figure=fig4)
ax1 = fig4.add_subplot(gs[0, 0:2])
ax2 = fig4.add_subplot(gs[0, 2:])
# ax3 = fig4.add_subplot(gs[1, 0:2])
# ax4 = fig4.add_subplot(gs[1, 2:])
ax3 = fig4.add_subplot(gs[1, 0:2])
ax4 = fig4.add_subplot(gs[1, 2])
ax5 = fig4.add_subplot(gs[1, 3])
qc_QFT.draw(output="mpl", style="iqp", ax=ax1)
plot_info_latt(info_latt_QFT, ax=ax2)
# qc_QFT2.draw(output="mpl", style="iqp", ax=ax3)
# plot_info_latt(info_latt_QFT2, ax=ax4)
qc_QFT3.draw(output="mpl", style="iqp", ax=ax3)
plot_info_latt(info_latt_QFT3_1, ax=ax4)
plot_info_latt(info_latt_QFT3_2, ax=ax5)
# fig4.suptitle('Information profile of c-phase gates used for the QFT')
fig4.savefig('qft-info.pdf', format='pdf', backend='pgf')
plt.show()
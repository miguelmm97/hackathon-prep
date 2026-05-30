#%% Imports

# Built-in modules
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Information lattice
from InfoLattice import calc_info, plot_info_latt


#%% Functions
def grover_oracle(marked_states):

    if not isinstance(marked_states, list):
        raise TypeError('marked_states should be a list of states in the computational basis.')

    # Initialise the circuit
    num_qubits = len(marked_states[0])
    qc = QuantumCircuit(num_qubits)

    # Mark each target state in the input list
    for target in marked_states:

        # Flip target bit-string to match Qiskit bit-ordering
        rev_target = target[::-1]

        # Create circuit that maps the target states
        zero_indx = [ind for ind in range(num_qubits) if rev_target.startswith('0', ind)]
        qc.x(zero_indx)
        qc.compose(MCMT(ZGate(), num_qubits - 1, 1), inplace=True)
        qc.x(zero_indx)

    return qc

#%% Evolving quantum states through the circuit

# Circuit details
marked_states = ['1010000']
num_qubits = len(marked_states[0])
oracle = grover_oracle(marked_states)
grover_op = GroverOperator(oracle, insert_barriers=True)
optimal_iter = math.floor(math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2 ** grover_op.num_qubits))))

# Initial state
qc = QuantumCircuit(num_qubits)
psi0 = Statevector.from_label('0000000')
info0 = calc_info(psi0.data)

# First step: Hadamard transform
qc.h(range(num_qubits))
psi1 = psi0.evolve(qc)
info1 = calc_info(psi1.data)

# Second step: grover iteration 1
qc.compose(grover_op, inplace=True)
psi2 = psi0.evolve(qc)
info2 = calc_info(psi2.data)

# Third step: grover iteration 2
qc.compose(grover_op, inplace=True)
psi3 = psi0.evolve(qc)
info3 = calc_info(psi3.data)

# Third step: grover iteration 2
qc.compose(grover_op, inplace=True)
psi4 = psi0.evolve(qc)
info4 = calc_info(psi4.data)

# Third step: grover iteration 2
qc.compose(grover_op, inplace=True)
psi5 = psi0.evolve(qc)
info5 = calc_info(psi5.data)

# Third step: grover iteration 2
qc.compose(grover_op, inplace=True)
psi6 = psi0.evolve(qc)
info6 = calc_info(psi6.data)

qc.compose(grover_op, inplace=True)
psi7 = psi0.evolve(qc)
info7 = calc_info(psi7.data)

qc.compose(grover_op, inplace=True)
psi8 = psi0.evolve(qc)
info8 = calc_info(psi8.data)


#%% Figures
fig1 = plt.figure(figsize=(6, 8))
gs = GridSpec(2, 2, figure=fig1, wspace=0.5, hspace=0.5)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[1, 0])
ax4 = fig1.add_subplot(gs[1, 1])

plot_info_latt(info5, ax1)
ax1.set_title('Step 0')
plot_info_latt(info6, ax2)
ax2.set_title('Step 1')
plot_info_latt(info7, ax3)
ax3.set_title('Step 2')
plot_info_latt(info8, ax4)
ax4.set_title('Step 3')

fig2 = grover_op.decompose().draw(output="mpl", style="iqp")
ax2 = fig2.gca()
ax2.set_title('Grover iteration', fontsize=20)

fig3 = qc.draw(output="mpl", style="iqp")
ax3 = fig3.gca()
ax3.set_title('Full Grover circuit', fontsize=20)

plt.show()















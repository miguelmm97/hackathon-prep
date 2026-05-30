import numpy as np
from scipy.linalg import eigvalsh as scipy_eigvalsh
from numpy.linalg import eigvalsh as numpy_eigvalsh
import matplotlib.pyplot as plt
from numpy import pi

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Managing data
import h5py
import os


# Algorithm
def qft_circuit(num_qubits):
    circuit = QuantumCircuit(num_qubits)
    for n in range(num_qubits - 1, -1, -1):

        # Series of H and controlled phase gates on every qubit
        if n != 0:
            circuit.compose(qft_block(n, num_qubits), inplace=True)
            circuit.barrier()

        # H on the last qubit and swap operations
        else:
            circuit.h(0)
            circuit.barrier()
            for i in range(num_qubits - 1, int(np.floor(num_qubits / 2)) - 1, -1):
                if i != num_qubits - i - 1:
                    circuit.swap(i, num_qubits - i - 1)
                else:
                    pass
            return circuit
    return circuit

def qft_gate(num_qubits):
    circuit = QuantumCircuit(num_qubits)
    for n in range(num_qubits - 1, -1, -1):

        # Series of H and controlled phase gates on every qubit
        if n != 0:
            circuit.compose(qft_block(n, num_qubits), inplace=True)

        # H on the last qubit and swap operations
        else:
            circuit.h(0)
            for i in range(num_qubits - 1, int(np.floor(num_qubits / 2)) - 1, -1):
                if i != num_qubits - i - 1:
                    circuit.swap(i, num_qubits - i - 1)
                else:
                    pass
            return circuit.to_gate()
    return circuit.to_gate()


def qft_block(qubit, num_qubits):
    circuit = QuantumCircuit(num_qubits)
    circuit.h(qubit)
    # Repeated controlled phase gates on all remaining qubits
    for target in range(qubit - 1, -1, - 1):
        circuit.cp(pi / 2 ** (qubit - target), target, qubit)
    return circuit

def Umod_multi(num_qubits, power, U, name='U'):

    # Apply U repeated times
    circ = QuantumCircuit(num_qubits)
    for _ in range(2 ** power):
        circ.append(U, range(num_qubits))

    # Create gate out of the circuit
    U_multi = circ.to_gate()
    U_multi.name = f'${name}(2^{power})$'
    cU_multi = U_multi.control()
    return cU_multi


# Information lattice
def reshape_psi(psi, n, l):
    '''

    Parameters
    ----------
    psi : numpy array
        Vector of size 2**L where L is the number of sites.
    n : INT
        the leftmost site of the subset of interest A.
    l : INT
        The number of sites in the subset of interest A.

    Returns
    -------
    psi: numpy array (2**l,2**(L-l))
        Reshaped psi in a matrix psi_{(A),(\bar{A}) with (A) the combined
        indices in A, and similar for \bar{A}

    '''

    L = int(np.log2(psi.size))
    # psi = np.transpose(np.reshape(psi,L*[2]))     # transpose since the binary representation is in opposite order in the basis
    psi = np.reshape(psi, L * [2])
    psi = np.moveaxis(psi, list(range(n, n + l)), list(range(0, l)))
    psi = np.reshape(psi, (2 ** l, 2 ** (L - l)))

    return psi

def S_vN(psi):
    '''

    Parameters
    ----------
    psi : np.array 2**Na x 2**Nb
        The wave function. Assumed to have been reshpaed into a matrix with
        the correct subspace dimensions

    Returns
    -------
    S : FLOAT
        the von Neumann entanglement entropy

    '''

    sv = np.linalg.svd(psi, compute_uv=False)
    sv = sv[sv > 1e-16]
    sv = sv ** 2
    #print("sv", sv)
    return -np.sum(sv * np.log2(sv))

def calc_entropies(psi):
    '''

    Parameters
    ----------
    psi : Numpy Array 2**L
        The wave function.

    Returns
    -------
    SvN : dictionary of np.arrays
        SvN[l] is the set of entropies of subspaces with size l

    '''

    L = int(np.log2(psi.size))
    SvN = {}

    for l in range(1, L + 1):
        SvN[l] = []
        for n in range(L - l + 1):
            # print("l",l-1)
            # print("n",n)
            psi_r = reshape_psi(psi, n, l)
            subsystem_S_vN = S_vN(psi_r)
            SvN[l].append(subsystem_S_vN)
            # print("subsystem_S_vN", subsystem_S_vN)
        SvN[l] = np.array(SvN[l])
    return SvN

def calc_info(psi):
    '''

    Parameters
    ----------
    psi : Numpy Array 2**L
        The wavefunction.

    Returns
    -------
    info_latt: dictionary of np.arrays
        info_latt[l] is the info on scale l

    '''

    assert len(psi.shape) == 1
    L = int(np.log2(psi.size))
    SvN = calc_entropies(psi)
    info_latt = {}
    for l in range(1, L + 1):
        if l == 1:
            info_latt[l] = l - SvN[l]
        elif l == 2:
            info_latt[l] = 2 - l - SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:]
        else:
            info_latt[l] = -SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:] - SvN[l - 2][1:-1]

    return info_latt

def calc_info_per_scale(info_latt, bc='periodic'):
    '''


    Parameters
    ----------
    info_latt: dictionary of np.arrays
        info_latt[l] is the info on scale l

    Returns
    -------
    info_per_scale: np.array
        info_per_scale[l] is the info summed over all sites on scale l

    '''

    L = len(info_latt)
    info_per_scale = np.zeros((L,))
    # info_per_scale = {}

    if bc == 'periodic':
        for l in range(1, L//2 + 2):
            if l == 1:
                info_per_scale[l] = np.sum(info_latt[l])
            else:
                if L % 2 == 0 and l == L//2 + 1:
                    info_per_scale[l] = np.sum(info_latt[l])
                else:
                    info_per_scale[l] = np.sum(info_latt[l]) + np.sum(info_latt[L-l+2])
    elif bc == 'open':
        for l in range(1, L + 1):
            info_per_scale[l-1] = np.sum(info_latt[l])

    else:
        raise ValueError('boundary condition must be "periodic" or "open"')

    return info_per_scale

def plot_info_latt(info_latt, ax):
    color_map = plt.get_cmap("Oranges")
    L = max(info_latt.keys())
    r = 1/(4*L)
    for l in info_latt:
        for x in range(len(info_latt[l])):
            ax.add_artist(plt.Circle((x/L+l/(2*L), (l-0.5)/L), r, color=color_map(info_latt[l][x])))
    ax.set_xlim([-2*r,1])
    ax.set_ylim([-2*r,1+2*r])
    ax.set_aspect('equal')
    ax.axis('off')

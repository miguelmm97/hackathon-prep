import numpy as np
from scipy.linalg import eigvalsh as scipy_eigvalsh
from numpy.linalg import eigvalsh as numpy_eigvalsh
import matplotlib.pyplot as plt

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Managing data
import h5py
import os


# Algorithms
def deutsch_function(num_qubits):
    """
    Returns a Deutsch-Joza function circuit implementation that with 50% chance is constant or balanced.
    """

    info = 'balanced'
    qc = QuantumCircuit(num_qubits + 1)

    # Flip target qubit to generate any of the four cases of f(x)
    if np.random.randint(0, 2):
        qc.x(num_qubits)

    # Return constant circuit with 1/2 probability: All 1s if target is flipped or 0s otherwise
    if np.random.randint(0, 2):
        info = 'constant'
        return qc, info

    # Balanced circuit: 1) Choose half of the states in H
    half_states = np.random.choice(range(2 ** num_qubits), (2 ** num_qubits) // 2,  replace=False)

    # 2) Add X gates wherever a qubit is |1 >
    def add_cx(qc, bit_string):
        for qubit, bit in enumerate(reversed(bit_string)):
            if bit == "1": qc.x(qubit)
        return qc

    # 3) Apply controlled X operations for all states in the list
    for state in half_states:
        qc.barrier()
        qc = add_cx(qc, f'{state:0b}')
        qc.mcx(list(range(num_qubits)), num_qubits)
        qc = add_cx(qc, f'{state:0b}')

    qc.barrier()

    return qc, info

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

def median(x, P):

    acc_prob = 0
    for i in range(len(x)):
        acc_prob += P[i]
        if acc_prob > 1/2 and i > 0:
            if P[i] >= P[i - 1]:
                percent = 1 - P[i - 1] / P[i]
            else:
                percent = P[i] / P[i - 1]
            med = x[i-1] + percent * x[i]
            break
        elif acc_prob > 1/2 and i == 0:
            med = x[0]

    return med


# Managing data
def get_fileID(file_list, common_name='datafile'):
    expID = 0
    for file in file_list:
        if file.startswith(common_name) and file.endswith('.h5'):
            stringID = file.split(common_name)[1].split('.h5')[0]
            ID = int(stringID)
            expID = max(ID, expID)
    return expID + 1

def store_my_data(file, name, data):
    try:
        file.create_dataset(name=name, data=data)
    except Exception as ex:
        print(f'Failed to write {name} in {file} because of exception: {ex}')

def attr_my_data(dataset, attr_name, attr):
    try:
        dataset.attrs.create(name=attr_name, data=attr)
    except Exception as ex:
        print(f'Failed to write {attr_name} in {dataset} because of exception: {ex}')

def load_my_data(file_list, directory):
    # Generate a dict with 1st key for filenames, 2nd key for datasets in the files
    data_dict = {}

    # Load desired directory and list files in it
    for file in file_list:
        file_path = os.path.join(directory, file)
        data_dict[file] = {}

        with h5py.File(file_path, 'r') as f:
            for group in f.keys():
                try:
                    data_dict[file][group] = {}
                    for dataset in f[group].keys():
                        if isinstance(f[group][dataset][()], bytes):
                            data_dict[file][group][dataset] = f[group][dataset][()].decode()
                        else:
                            data_dict[file][group][dataset] = f[group][dataset][()]
                except AttributeError:
                    if isinstance(f[group][()], bytes):
                        data_dict[file][group] = f[group][()].decode()
                    else:
                        data_dict[file][group] = f[group][()]

    return data_dict

def load_my_attr(file_list, directory, dataset):
    attr_dict = {}

    # Load desired directory and list files in it
    for file in file_list:
        file_path = os.path.join(directory, file)
        attr_dict[file] = {}
        print(file)

        with h5py.File(file_path, 'r') as f:
            for att in f[dataset].attrs.keys():
                attr_dict[file][att] = f[dataset].attrs[att]

    return attr_dict



# Quantum Algorithms Information Lattice

This repository contains exploratory Python scripts for studying how information is distributed across qubit scales during standard quantum algorithms. It uses Qiskit statevector simulation to evolve quantum states through algorithm circuits, then computes and plots an entropy-based "information lattice" at selected stages of the computation.

The main algorithms covered are:

- Deutsch-Jozsa
- Grover search
- Quantum Fourier Transform (QFT)
- Shor-style period finding examples for multiplication modulo 15

The code is research/plot-generation oriented rather than a packaged Python library. Most scripts are intended to be run directly from their algorithm-specific directories.

## What The Code Does

For each algorithm, the scripts generally follow this workflow:

1. Build a quantum circuit using Qiskit.
2. Initialize a `Statevector`.
3. Evolve the state through selected gates or algorithm steps.
4. Compute subsystem von Neumann entropies.
5. Combine those entropies into an information lattice by qubit scale.
6. Plot the circuit and/or information lattice using Matplotlib.
7. In some Grover scripts, save aggregate information-scale data to HDF5 files for later post-processing.

The information lattice utilities are implemented in each algorithm folder's `functions.py`. These utilities reshape a statevector into subsystem/environment partitions, compute singular values, calculate entanglement entropy, and derive information contributions by scale.

## Repository Structure

```text
algorithms/
├── README.md
├── deutsch-algorithm/
│   ├── functions.py
│   └── info-deutsch-general.py
├── grover-algorithm/
│   ├── functions.py
│   ├── info-grover-general.py
│   ├── info-grover-example.py
│   ├── grover-postprod.py
│   ├── grover.pdf
│   ├── mean_scale.pdf
│   └── median_scale.pdf
└── schor-algorithm/
    ├── functions.py
    ├── qft-example.py
    ├── info-qft-general.py
    ├── shor-example.py
    ├── info-shor-general-2mod15.py
    ├── info-shor-general-7mod15.py
    ├── cphase-swaps-U-infoprofile.py
    └── generated PDF figures
```

### `deutsch-algorithm`

Contains a Deutsch-Jozsa example. The main script is:

- `info-deutsch-general.py`: builds a Deutsch-Jozsa circuit with a randomly generated constant or balanced oracle, computes information lattices at each major step, and plots the results.

### `grover-algorithm`

Contains Grover search simulations and post-processing.

- `info-grover-general.py`: runs Grover search for a chosen initial state and marked state, tracks the information lattice over Grover iterations, computes information per scale, saves HDF5 data, and generates figures.
- `grover-postprod.py`: loads saved Grover experiment files and plots mean/median information scale across runs.
- `info-grover-example.py`: older/manual Grover example. Note that it imports `InfoLattice`, which is not present in this directory; prefer `info-grover-general.py` unless this import is restored or updated.

### `schor-algorithm`

Contains QFT and Shor-style period-finding examples.

- `qft-example.py`: builds and saves a QFT circuit diagram.
- `info-qft-general.py`: tracks the information lattice through individual QFT blocks.
- `shor-example.py`: builds a Shor-style circuit for multiplication by `7 mod 15`.
- `info-shor-general-2mod15.py`: tracks information through a Shor-style circuit using multiplication by `2 mod 15`.
- `info-shor-general-7mod15.py`: tracks information through a Shor-style circuit using multiplication by `7 mod 15`.
- `cphase-swaps-U-infoprofile.py`: studies information profiles for controlled-phase gates, SWAP gates, modular gates, and QFT components.

## Dependencies

The scripts require Python 3 and the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `qiskit`
- `h5py`

Some scripts save figures using Matplotlib's PGF backend and enable LaTeX rendering with:

```python
plt.rc('text', usetex=True)
```

For those figure exports to work, a working LaTeX installation may be required. If LaTeX is not installed, either install a TeX distribution or disable `usetex=True` / change the save backend in the relevant script.

## Quick Start

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy scipy matplotlib seaborn qiskit h5py
```

Run an example:

```bash
cd deutsch-algorithm
python info-deutsch-general.py
```

Run a QFT example:

```bash
cd schor-algorithm
python qft-example.py
```

Run a Shor-style information-lattice example:

```bash
cd schor-algorithm
python info-shor-general-7mod15.py
```

Run the main Grover simulation:

```bash
cd grover-algorithm
python info-grover-general.py
```

The Grover script writes HDF5 output to `../../data-grover`. Create that directory first if it does not exist:

```bash
mkdir -p ../../data-grover
```

## Notes For New Users

- Run scripts from inside their own algorithm folder so that local imports like `from functions import ...` resolve correctly.
- Generated PDF figures are committed in the algorithm folders and can be regenerated by running the corresponding scripts.
- The project does not currently include a `requirements.txt`, test suite, or package entry point.
- Several helper functions are duplicated across the algorithm folders. If the project grows, these could be moved into a shared module.
- Some scripts are exploratory and may need small path/import updates depending on your environment.

## Typical Outputs

Depending on the script, outputs include:

- Qiskit circuit diagrams.
- Information lattice plots.
- PDF figures such as `grover.pdf`, `qft.pdf`, `shor.pdf`, `2mod15-info.pdf`, and `7mod15-info.pdf`.
- HDF5 experiment files for Grover simulations.

## Suggested Development Improvements

Useful next steps for making this easier to maintain:

- Add a `requirements.txt` or `pyproject.toml`.
- Move shared information-lattice functions into one common module.
- Replace relative data paths with configurable paths.
- Add a small smoke-test script that imports the helpers and runs one minimal circuit.
- Update older scripts that import missing modules, such as `InfoLattice`.

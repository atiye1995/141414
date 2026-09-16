"""
STEP 1 (per system) -- Build H4: linear chain, real Hamiltonian, VQE,
noise-response table. Takes ~15-30s (18-parameter VQE optimization).
"""
import pickle
import numpy as np
from physics_engine import (build_molecular_hamiltonian, qubit_op_to_pauli_dict,
                             hf_bitstring_state, build_double_excitation_generators,
                             run_vqe)
from openfermion.linalg import get_sparse_operator
from noise_model import build_q_true_table

geometry = [('H', (0, 0, 0)), ('H', (0, 0, 1.4)), ('H', (0, 0, 2.8)), ('H', (0, 0, 4.2))]
n_qubits = 8
n_electrons = 4

qubit_op, mol = build_molecular_hamiltonian(geometry)
pauli_terms = qubit_op_to_pauli_dict(qubit_op, n_qubits)
print(f"H4: n_pauli={len(pauli_terms)}  E_fci={mol.fci_energy:.8f}")

H_sparse = get_sparse_operator(qubit_op, n_qubits=n_qubits)
dim = 2 ** n_qubits
psi0, _ = hf_bitstring_state(n_qubits, n_electrons, dim)
gens, excs = build_double_excitation_generators(n_qubits, n_electrons, H_sparse)
print(f"H4: {len(gens)} VQE parameters")
psi_opt, E_vqe = run_vqe(H_sparse, psi0, gens)
print(f"H4: E_VQE={E_vqe:.8f}  |E_VQE-E_FCI|={abs(E_vqe-mol.fci_energy)*1000:.4f} mHartree")

with open('H4_pauli_terms.pkl', 'wb') as f:
    pickle.dump(pauli_terms, f)
np.save('H4_psi_opt.npy', psi_opt)
np.save('H4_E_fci.npy', np.array([mol.fci_energy]))

labels = sorted(pauli_terms.keys())
q_table = build_q_true_table(labels, psi_opt, n_qubits, channel='ampdamp')
np.save('H4_q_table.npy', q_table)
print("H4: done. Files saved: H4_pauli_terms.pkl, H4_psi_opt.npy, H4_E_fci.npy, H4_q_table.npy")

"""
STEP 1 (per system) -- Build LiH: frozen-core active space (2 electrons,
5 active orbitals including Li 2p), real Hamiltonian, VQE, noise-response
table. Takes ~10-15s.
"""
import pickle
import numpy as np
from physics_engine import (build_molecular_hamiltonian, qubit_op_to_pauli_dict,
                             hf_bitstring_state, build_double_excitation_generators,
                             run_vqe)
from openfermion.linalg import get_sparse_operator
from noise_model import build_q_true_table

BOHR = 0.529177210903
geometry = [('Li', (0, 0, 0)), ('H', (0, 0, 1.5949 / BOHR))]
n_qubits = 10
n_electrons = 2  # active electrons only (frozen Li 1s core)

qubit_op, mol = build_molecular_hamiltonian(
    geometry, occupied_indices=[0], active_indices=[1, 2, 3, 4, 5])
pauli_terms = qubit_op_to_pauli_dict(qubit_op, n_qubits)
print(f"LiH: n_pauli={len(pauli_terms)}")

H_sparse = get_sparse_operator(qubit_op, n_qubits=n_qubits)
dim = 2 ** n_qubits
psi0, _ = hf_bitstring_state(n_qubits, n_electrons, dim)
gens, excs = build_double_excitation_generators(n_qubits, n_electrons, H_sparse)
print(f"LiH: {len(gens)} VQE parameters")
psi_opt, E_vqe = run_vqe(H_sparse, psi0, gens)

from scipy.sparse.linalg import eigsh
E_fci_active = eigsh(H_sparse, k=1, which='SA', return_eigenvectors=False)[0]
print(f"LiH: E_fci(active space)={E_fci_active:.8f}  E_VQE={E_vqe:.8f}  "
      f"|E_VQE-E_FCI|={abs(E_vqe-E_fci_active)*1000:.4f} mHartree")

with open('LiH_pauli_terms.pkl', 'wb') as f:
    pickle.dump(pauli_terms, f)
np.save('LiH_psi_opt.npy', psi_opt)
np.save('LiH_E_fci.npy', np.array([E_fci_active]))

labels = sorted(pauli_terms.keys())
q_table = build_q_true_table(labels, psi_opt, n_qubits, channel='ampdamp')
np.save('LiH_q_table.npy', q_table)
print("LiH: done. Files saved: LiH_pauli_terms.pkl, LiH_psi_opt.npy, LiH_E_fci.npy, LiH_q_table.npy")

"""
CORE PHYSICS ENGINE - shared by H2, H4, LiH.
Uses OpenFermion + PySCF (validated to machine precision against
independent hand-rolled code in the original project) for the
Hamiltonian, and a generic double-excitation UCC-style VQE ansatz
that works for any (n_qubits, n_electrons).
"""
import numpy as np
from scipy.sparse.linalg import expm_multiply, eigsh
from scipy.sparse import csc_matrix
from scipy.optimize import minimize
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import run_pyscf
import pickle
import time

BOHR = 0.529177210903


def build_molecular_hamiltonian(geometry_bohr, basis='sto-3g', charge=0,
                                 multiplicity=1, occupied_indices=None,
                                 active_indices=None):
    """geometry_bohr: list of (symbol, (x,y,z)) in BOHR units."""
    geometry_ang = [(sym, tuple(c * BOHR for c in xyz)) for sym, xyz in geometry_bohr]
    mol = MolecularData(geometry_ang, basis, multiplicity=multiplicity, charge=charge)
    mol = run_pyscf(mol, run_scf=True, run_fci=True)
    ham = mol.get_molecular_hamiltonian(occupied_indices=occupied_indices,
                                         active_indices=active_indices)
    fermion_op = get_fermion_operator(ham)
    qubit_op = jordan_wigner(fermion_op)
    return qubit_op, mol


def qubit_op_to_pauli_dict(qubit_op, n_qubits):
    """Convert OpenFermion QubitOperator to {label_string: coeff} dict,
    with label[0] = leftmost qubit, matching this project's convention."""
    pauli_terms = {}
    for term, coeff in qubit_op.terms.items():
        label = ['I'] * n_qubits
        for (qi, pc) in term:
            label[qi] = pc
        lbl = ''.join(label)
        pauli_terms[lbl] = pauli_terms.get(lbl, 0.0) + coeff.real
    return pauli_terms


def hf_bitstring_state(n_qubits, n_electrons, dim):
    """HF reference: lowest n_electrons spin-orbitals occupied."""
    idx = 0
    for i in range(n_electrons):
        idx |= (1 << (n_qubits - 1 - i))
    psi = np.zeros(dim, dtype=complex)
    psi[idx] = 1.0
    return psi, idx


def build_double_excitation_generators(n_qubits, n_electrons, H_sparse):
    """Generic paired double-excitation generators: occ alpha/beta <-> virt
    alpha/beta pairs, built directly as sparse operators via OpenFermion-style
    JW sparse matrices for a_p^dagger a_q. Returns list of sparse anti-Hermitian
    generators."""
    from openfermion.ops import FermionOperator
    from openfermion.transforms import jordan_wigner as jw
    from openfermion.linalg import get_sparse_operator as gso

    occ = list(range(n_electrons))
    virt = list(range(n_electrons, n_qubits))
    occ_alpha = [p for p in occ if p % 2 == 0]
    occ_beta = [p for p in occ if p % 2 == 1]
    virt_alpha = [p for p in virt if p % 2 == 0]
    virt_beta = [p for p in virt if p % 2 == 1]

    excitations = []
    # alpha-alpha pairs
    for i in range(len(occ_alpha)):
        for j in range(i + 1, len(occ_alpha)):
            for a in range(len(virt_alpha)):
                for b in range(a + 1, len(virt_alpha)):
                    excitations.append((occ_alpha[i], occ_alpha[j], virt_alpha[a], virt_alpha[b]))
    # beta-beta pairs
    for i in range(len(occ_beta)):
        for j in range(i + 1, len(occ_beta)):
            for a in range(len(virt_beta)):
                for b in range(a + 1, len(virt_beta)):
                    excitations.append((occ_beta[i], occ_beta[j], virt_beta[a], virt_beta[b]))
    # mixed alpha-beta pairs
    for oa in occ_alpha:
        for ob in occ_beta:
            for va in virt_alpha:
                for vb in virt_beta:
                    excitations.append((oa, ob, va, vb))

    generators = []
    for (p, q, r, s) in excitations:
        op = FermionOperator(f'{r}^ {s}^ {q} {p}') - FermionOperator(f'{p}^ {q}^ {s} {r}')
        gen_sparse = gso(jw(op), n_qubits=n_qubits)
        generators.append(gen_sparse)
    return generators, excitations


def run_vqe(H_sparse, psi0, generators, maxiter_powell=3000):
    """Generic VQE: optimize sum(theta_i * generator_i) applied to psi0."""
    n_params = len(generators)

    def energy(thetas):
        state = psi0.copy()
        for th, G in zip(thetas, generators):
            if abs(th) > 1e-12:
                state = expm_multiply(th * G, state)
        state = state / np.linalg.norm(state)
        return np.real(np.vdot(state, H_sparse.dot(state)))

    theta0 = np.zeros(n_params)
    t0 = time.time()
    res = minimize(energy, theta0, method='Powell',
                    options={'xtol': 1e-7, 'ftol': 1e-10, 'maxiter': maxiter_powell})
    res2 = minimize(energy, res.x, method='BFGS', options={'gtol': 1e-9, 'maxiter': 1000})
    best = res2 if res2.fun < res.fun else res
    print(f"  VQE: {n_params} params, E={best.fun:.8f}, time={time.time()-t0:.1f}s")

    state = psi0.copy()
    for th, G in zip(best.x, generators):
        if abs(th) > 1e-12:
            state = expm_multiply(th * G, state)
    state = state / np.linalg.norm(state)
    return state, best.fun


if __name__ == "__main__":
    # Quick self-test on H2
    geometry = [('H', (0, 0, 0)), ('H', (0, 0, 1.4))]
    qubit_op, mol = build_molecular_hamiltonian(geometry)
    n_qubits = 4
    pauli_terms = qubit_op_to_pauli_dict(qubit_op, n_qubits)
    print("H2 test: n_pauli =", len(pauli_terms), " FCI =", mol.fci_energy)
    H_sparse = get_sparse_operator(qubit_op, n_qubits=n_qubits)
    dim = 2 ** n_qubits
    psi0, hf_idx = hf_bitstring_state(n_qubits, 2, dim)
    E_hf = np.real(np.vdot(psi0, H_sparse.dot(psi0)))
    print("H2 test: E_HF =", E_hf)
    gens, excs = build_double_excitation_generators(n_qubits, 2, H_sparse)
    print("H2 test: n_generators =", len(gens))
    psi_opt, E_vqe = run_vqe(H_sparse, psi0, gens)
    print("H2 test: E_VQE =", E_vqe, " |E_VQE-E_FCI| =", abs(E_vqe - mol.fci_energy))

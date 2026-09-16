"""
NOISE MODEL - closed-form Pauli-transfer approach (validated to machine
precision against full Kraus density-matrix simulation in the original
project). Much faster than building dense channels: O(n_pauli * 2^{n_Z})
instead of O(n_pauli * dim^2) per noise-scale evaluation.
"""
import numpy as np


def apply_pauli_to_statevector(label, psi, n_qubits):
    dim = len(psi)
    out = np.zeros(dim, dtype=complex)
    nz = np.nonzero(psi)[0]
    for i in nz:
        amp = psi[i]
        new_idx = int(i)
        phase = 1.0 + 0.0j
        for q, ch in enumerate(label):
            bit = (new_idx >> (n_qubits - 1 - q)) & 1
            if ch == 'I':
                continue
            elif ch == 'Z':
                phase *= (1 if bit == 0 else -1)
            elif ch == 'X':
                new_idx ^= (1 << (n_qubits - 1 - q))
            elif ch == 'Y':
                phase *= (1j if bit == 0 else -1j)
                new_idx ^= (1 << (n_qubits - 1 - q))
        out[new_idx] += phase * amp
    return out


def pauli_expectation(label, psi, n_qubits):
    Ppsi = apply_pauli_to_statevector(label, psi, n_qubits)
    return np.real(np.vdot(psi, Ppsi))


def build_ampdamp_cache(pauli_labels, psi, n_qubits):
    """Precompute <P_T> for all Z-subterms needed by the amplitude/phase
    damping closed form: <P>(s) = kappa(s)^n_xy * sum_T (1-gamma)^|T| gamma^(nz-|T|) <P_T>_0."""
    cache = {}
    for label in pauli_labels:
        z_positions = [i for i, c in enumerate(label) if c == 'Z']
        n_xy = sum(1 for c in label if c in ('X', 'Y'))
        n_z = len(z_positions)
        sub_expectations = {}
        for mask in range(2 ** n_z):
            T = frozenset(z_positions[i] for i in range(n_z) if (mask >> i) & 1)
            sub_label = list(label)
            for zp in z_positions:
                if zp not in T:
                    sub_label[zp] = 'I'
            sub_label = ''.join(sub_label)
            sub_expectations[T] = pauli_expectation(sub_label, psi, n_qubits)
        cache[label] = dict(z_positions=z_positions, n_xy=n_xy, sub_expectations=sub_expectations)
    return cache


def predict_expectation_ampdamp(label_cache, gamma0, gamma_phi0, s):
    gamma = min(gamma0 * s, 1.0)
    lam_ph = min(gamma_phi0 * s, 1.0)
    kappa = np.sqrt(max(1 - gamma, 0)) * np.sqrt(max(1 - lam_ph, 0))
    n_xy = label_cache['n_xy']
    total = 0.0
    for T, val in label_cache['sub_expectations'].items():
        n_kept = len(T)
        n_excl = len(label_cache['z_positions']) - n_kept
        coeff = (1 - gamma) ** n_kept * gamma ** n_excl
        total += coeff * val
    return (kappa ** n_xy) * total


def predict_expectation_depolarizing(label, exp0, p0, s):
    weight = sum(1 for c in label if c != 'I')
    p = min(p0 * s, 1.0)
    return (1 - p) ** weight * exp0


def build_q_true_table(pauli_labels, psi, n_qubits, lambdas=(0.5, 1.0, 1.5),
                        channel='ampdamp', gamma0=0.02, gamma_phi0=0.03, p0=0.025):
    """Returns array shape (len(lambdas), len(pauli_labels)) of q_i(lambda)=(1+<P>)/2."""
    n_pauli = len(pauli_labels)
    q_table = np.zeros((len(lambdas), n_pauli))
    if channel == 'ampdamp':
        cache = build_ampdamp_cache(pauli_labels, psi, n_qubits)
        for il, s in enumerate(lambdas):
            for ip, label in enumerate(pauli_labels):
                P_exp = predict_expectation_ampdamp(cache[label], gamma0, gamma_phi0, s)
                q_table[il, ip] = (1 + P_exp) / 2
    elif channel == 'depolarizing':
        exp0_all = {label: pauli_expectation(label, psi, n_qubits) for label in pauli_labels}
        for il, s in enumerate(lambdas):
            for ip, label in enumerate(pauli_labels):
                P_exp = predict_expectation_depolarizing(label, exp0_all[label], p0, s)
                q_table[il, ip] = (1 + P_exp) / 2
    else:
        raise ValueError(channel)
    return np.clip(q_table, 0.0, 1.0)


if __name__ == "__main__":
    import pickle, time
    with open('H4_pauli_terms.pkl', 'rb') as f:
        pauli_terms = pickle.load(f)
    psi = np.load('H4_psi_opt.npy')
    labels = list(pauli_terms.keys())
    t0 = time.time()
    q_table = build_q_true_table(labels, psi, 8, channel='ampdamp')
    print(f"H4 q_true_table built in {time.time()-t0:.2f}s, shape={q_table.shape}")
    print("Sample values:", q_table[:, :3])

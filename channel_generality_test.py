"""Tests r_true under both noise channels, all three systems -- fast,
pure physics (no statistical sampling needed)."""
import numpy as np, pickle
from noise_model import pauli_expectation, predict_expectation_depolarizing, build_ampdamp_cache, predict_expectation_ampdamp

def compute_r_true(pauli_terms, psi, n_qubits, channel, **kwargs):
    labels = sorted(pauli_terms.keys())
    coeffs = np.array([pauli_terms[l] for l in labels])
    lambdas = [0.5, 1.0, 1.5]
    Es = {}
    if channel == 'ampdamp':
        cache = build_ampdamp_cache(labels, psi, n_qubits)
        for s in lambdas:
            q = np.array([(1+predict_expectation_ampdamp(cache[l], kwargs['gamma0'], kwargs['gamma_phi0'], s))/2 for l in labels])
            Es[s] = np.sum(coeffs*(2*q-1))
    else:
        exp0 = {l: pauli_expectation(l, psi, n_qubits) for l in labels}
        for s in lambdas:
            q = np.array([(1+predict_expectation_depolarizing(l, exp0[l], kwargs['p0'], s))/2 for l in labels])
            Es[s] = np.sum(coeffs*(2*q-1))
    r = (Es[1.5]-Es[1.0])/(Es[1.0]-Es[0.5])
    return r

results = {}
for name, nq in [('H2',4),('H4',8),('LiH',10)]:
    with open(f'{name}_pauli_terms.pkl','rb') as f: pt = pickle.load(f)
    psi = np.load(f'{name}_psi_opt.npy')
    r_dep = compute_r_true(pt, psi, nq, 'depolarizing', p0=0.025)
    r_amp = compute_r_true(pt, psi, nq, 'ampdamp', gamma0=0.02, gamma_phi0=0.03)
    results[name] = (r_dep, r_amp)
    print(f'{name}: r_true(depolarizing)={r_dep:.4f}  r_true(amp+phase damping)={r_amp:.4f}')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
systems = list(results.keys())
r_dep = [results[s][0] for s in systems]
r_amp = [results[s][1] for s in systems]
x = np.arange(len(systems)); width=0.35
fig, ax = plt.subplots(figsize=(7,4.5))
ax.bar(x-width/2, r_dep, width, label='Depolarizing (unital)', color='#4C72B0')
ax.bar(x+width/2, r_amp, width, label='Amplitude+phase damping (non-unital)', color='#C44E52')
ax.axhline(1.0, color='black', linestyle=':', linewidth=1.2)
ax.set_xticks(x); ax.set_xticklabels(systems)
ax.set_ylabel('r_true'); ax.set_ylim(0.95,1.05)
ax.set_title('Reproduced: r_true \u2248 1 independent of channel type')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('chart_channel_generality_reproduced.png', dpi=150)
print("saved chart_channel_generality_reproduced.png")

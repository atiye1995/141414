"""
Runs the linear-only vs BMA-Laplace vs Spike-and-Slab comparison for all
three systems and saves a summary CSV + the main comparison chart.
"""
import numpy as np
import pickle
import pandas as pd
import time
from stats_analysis import *

SYSTEMS = {
    'H2':  dict(n_qubits=4,  Arange=(-3, 1),  E_fci=-1.1372759436172823, n_shots_list=[1000, 500, 200], n_reps=150),
    'H4':  dict(n_qubits=8,  Arange=(-3, 1),  E_fci=-2.139442549044604,  n_shots_list=[1000, 500, 200], n_reps=150),
    'LiH': dict(n_qubits=10, Arange=(-10,-6), E_fci=-7.882175990801272,  n_shots_list=[1000, 500, 200], n_reps=100),
}

records = []
for sysname, cfg in SYSTEMS.items():
    with open(f'{sysname}_pauli_terms.pkl', 'rb') as f:
        pt = pickle.load(f)
    labels = sorted(pt.keys())
    coeffs = np.array([pt[l] for l in labels])
    q_table = np.load(f'{sysname}_q_table.npy')
    E_fci = cfg['E_fci']

    for n_shots in cfg['n_shots_list']:
        t0 = time.time()
        inside_lin = inside_bma = inside_ss = 0
        w_lin, w_bma, w_ss = [], [], []
        for rep in range(cfg['n_reps']):
            rng = np.random.default_rng(hash((sysname, n_shots, rep)) % (2**32))
            counts = rng.binomial(n_shots, q_table)

            lo, hi = linear_only_interval(counts, n_shots, coeffs)
            w_lin.append(hi - lo); inside_lin += (lo <= E_fci <= hi)

            res = bma_laplace_interval(counts, n_shots, coeffs, rng, Arange=cfg['Arange'], n_mix=8000)
            if res is not None:
                lo2, hi2 = res
                w_bma.append(hi2 - lo2); inside_bma += (lo2 <= E_fci <= hi2)

            theta_scale = calibrate_theta_scale(counts, n_shots)
            lo3, hi3, p_spike = spike_slab_interval(counts, n_shots, coeffs, rng, cfg['Arange'], theta_scale)
            w_ss.append(hi3 - lo3); inside_ss += (lo3 <= E_fci <= hi3)

        n_bma = len(w_bma)
        records.append(dict(system=sysname, n_shots=n_shots, method='linear_only',
                             coverage=inside_lin/cfg['n_reps']*100, width_mHa=np.median(w_lin)*1000))
        records.append(dict(system=sysname, n_shots=n_shots, method='bma_laplace',
                             coverage=inside_bma/max(n_bma,1)*100, width_mHa=np.median(w_bma)*1000 if w_bma else np.nan))
        records.append(dict(system=sysname, n_shots=n_shots, method='spike_slab',
                             coverage=inside_ss/cfg['n_reps']*100, width_mHa=np.median(w_ss)*1000))
        print(f"{sysname} n_shots={n_shots}: done in {time.time()-t0:.1f}s")

df = pd.DataFrame(records)
df.to_csv('comparison_summary.csv', index=False)
print("\n", df.to_string(index=False))

import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv('comparison_summary.csv')
systems = ['H2','H4','LiH']
methods = ['linear_only','bma_laplace','spike_slab']
labels_fa = {'linear_only':'Linear-only','bma_laplace':'BMA-Laplace','spike_slab':'Spike-and-Slab'}
colors = {'linear_only':'#8172B2','bma_laplace':'#DD8452','spike_slab':'#55A868'}

fig, axes = plt.subplots(1,3, figsize=(15,4.5))
for ax, sysname in zip(axes, systems):
    shots = [1000,500,200]
    x = np.arange(len(shots))
    width=0.25
    for i,m in enumerate(methods):
        sub = df[(df.system==sysname)&(df.method==m)].set_index('n_shots').loc[shots]
        ax.bar(x+i*width-width, sub.width_mHa, width=width, label=labels_fa[m], color=colors[m])
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in shots])
    ax.set_title(sysname); ax.set_xlabel('n_shots')
    ax.set_yscale('log')
    if ax is axes[0]: ax.set_ylabel('Median CI width (mHartree, log scale)')
    ax.legend(fontsize=8)
plt.suptitle('Reproduced from scratch: Linear-only vs BMA vs Spike-and-Slab (all 3 systems)')
plt.tight_layout()
plt.savefig('chart_comparison_reproduced.png', dpi=150)
print("saved chart_comparison_reproduced.png")

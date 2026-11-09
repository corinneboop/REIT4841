### STEP 3 - Collinearity: which variables carry the same information?

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
SOCIAL_DIR = os.path.dirname(HERE)
sys.path.insert(0, SOCIAL_DIR)
from variable_sets import REFERENCE_OMIT

OUT = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(SOCIAL_DIR, '2_analysis_dataset', 'output', 'analysis_dataset.csv')
VDICT = os.path.join(SOCIAL_DIR, '2_analysis_dataset', 'output', 'variable_dictionary.csv')
HIGH = 0.7
VIF_LIMIT = 5
PROTECTED = ['log_income']
CONTEXT = ['business_share_pct', 'res_kWh_per_customer_day', 'log_bus_customers_per_1000_dw',
           'industrial_substation_pct', 'log_dwelling_density', 'res_tr_kVA_per_dwelling', 'log_ceg_per_solar_kVA',
           'pv_kVA_per_dwelling', 'avg_solar_kVA_per_site', 'major_city', 'dist_coast_km']

d = pd.read_csv(DATA, dtype={'POSTCODE': str}).set_index('POSTCODE')
vd = pd.read_csv(VDICT).set_index('variable')
LABELS = vd['description'].to_dict()
X_ALL = [v for v in vd.index if v not in CONTEXT]
X_MULTI = [v for v in X_ALL if v not in REFERENCE_OMIT]
print(f"{len(d)} postcodes, {len(X_ALL)} social variables ({len(X_MULTI)} in the multivariable set; reference left out: "
      f"{', '.join(REFERENCE_OMIT)})")

corr = d[X_ALL].corr(method='spearman')
corr.round(3).to_csv(os.path.join(OUT, '3_spearman_correlation.csv'))
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(X_ALL) for b in X_ALL[i + 1:] if abs(corr.loc[a, b]) >= HIGH]
pairs = pd.DataFrame(pairs, columns=['variable_1', 'variable_2', 'spearman_rho'])
pairs = pairs.reindex(pairs['spearman_rho'].abs().sort_values(ascending=False).index)
pairs['description_1'] = pairs['variable_1'].map(LABELS); pairs['description_2'] = pairs['variable_2'].map(LABELS)
pairs.round(3).to_csv(os.path.join(OUT, '3_high_correlation_pairs.csv'), index=False)

parent = {v: v for v in X_ALL}
def find(v):
    while parent[v] != v:
        v = parent[v]
    return v
for a, b in pairs[['variable_1', 'variable_2']].itertuples(index=False, name=None):
    parent[find(a)] = find(b)
groups = pd.Series({v: find(v) for v in X_ALL})
clusters = []
for k, (root, members) in enumerate(sorted(groups.groupby(groups), key=lambda g: -len(g[1])), start=1):
    m = members.index.tolist()
    for v in m:
        clusters.append({'cluster': k, 'size': len(m), 'variable': v, 'description': LABELS[v],
                         })
clusters = pd.DataFrame(clusters)
clusters.round(3).to_csv(os.path.join(OUT, '3_collinear_clusters.csv'), index=False)

def vifs(cols):
    Xc = sm.add_constant(d[cols])
    return pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])], index=cols)

vif_full = vifs(X_MULTI).sort_values(ascending=False)
pd.DataFrame({'description': vif_full.index.map(LABELS), 'VIF': vif_full}).round(2).to_csv(
    os.path.join(OUT, '3_vif_full.csv'), index_label='variable')

cols, steps = list(X_MULTI), []
while True:
    v = vifs(cols)
    cand = v.drop(PROTECTED, errors='ignore')
    if v.max() < VIF_LIMIT or cand.empty:
        break
    drop = cand.idxmax()
    cols.remove(drop)
    steps.append({'step': len(steps) + 1, 'removed': drop, 'description': LABELS[drop], 'VIF_when_removed': v[drop],
                  'max_VIF_after': vifs(cols).max(), 'variables_left': len(cols)})
steps = pd.DataFrame(steps)
kept = cols
rep = {}
for s in steps['removed']:
    best = corr.loc[s, kept].abs().idxmax()
    rep[s] = (best, corr.loc[s, best])
steps['most_correlated_kept_variable'] = steps['removed'].map(lambda s: rep[s][0])
steps['rho_with_it'] = steps['removed'].map(lambda s: rep[s][1])
steps.round(3).to_csv(os.path.join(OUT, '3_vif_trimming_steps.csv'), index=False)
vk = vifs(kept)
kt = pd.DataFrame({'variable': kept, 'description': [LABELS[v] for v in kept], 'VIF': vk.values,
                   'also_represents': [', '.join(f"{s} ({rep[s][1]:+.2f})" for s in rep if rep[s][0] == v) for v in kept]})
kt.round(3).to_csv(os.path.join(OUT, '3_kept_variables.csv'), index=False)

pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 90)
print(f"\nHighly collinear pairs (|rho| >= {HIGH}): {len(pairs)}")
print(pairs[['variable_1', 'variable_2', 'spearman_rho']].round(2).to_string(index=False))
print('\nClusters (linked by |rho| >= 0.7):')
for k, g in clusters.groupby('cluster'):
    if len(g) > 1:
        print(f"  {k}: " + ', '.join(g['variable']))
print('\nVIF, all variables:'); print(vif_full.round(1).to_string())
print('\nVIF trimming:'); print(steps[['removed', 'VIF_when_removed', 'max_VIF_after', 'most_correlated_kept_variable', 'rho_with_it']].round(2).to_string(index=False))
print('\nKept:'); print(kt[['variable', 'VIF', 'also_represents']].round(2).to_string(index=False))

dist = 1 - corr.abs().values
np.fill_diagonal(dist, 0)
order = [X_ALL[i] for i in leaves_list(linkage(squareform(dist, checks=False), 'average'))]
cm = corr.loc[order, order]
fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), gridspec_kw={'width_ratios': [1.35, 0.65]})
ax = axes[0]
im = ax.imshow(cm.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(order))); ax.set_xticklabels([LABELS[v] for v in order], rotation=90, fontsize=8)
ax.set_yticks(range(len(order))); ax.set_yticklabels([LABELS[v] for v in order], fontsize=8)
for i in range(len(order)):
    for j in range(len(order)):
        val = cm.values[i, j]
        if i != j and abs(val) >= HIGH:
            ax.text(j, i, f'{val:.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=6.5,
                    color='white' if abs(val) > 0.75 else 'black')
fig.colorbar(im, ax=ax, shrink=0.7, label='Spearman rho')
ax.set_title(f'(a) Correlation between variables, ordered by similarity\n(numbers shown where |rho| >= {HIGH})',
             loc='left', fontsize=10)
ax = axes[1]
vv = vif_full.sort_values()
ax.barh(range(len(vv)), vv.values, color=['#2a78d6' if v in kept else '#c4c4c4' for v in vv.index])
ax.axvline(VIF_LIMIT, color='#e34948', lw=1, ls='--'); ax.axvline(10, color='#e34948', lw=1, ls=':')
ax.set_yticks(range(len(vv))); ax.set_yticklabels([LABELS[v] for v in vv.index], fontsize=8)
ax.set_xscale('log'); ax.set_xlabel('VIF with all other variables (log scale); dashed = 5, dotted = 10')
ax.set_title('(b) Variance inflation factor\n(blue = kept after trimming to VIF < 5)', loc='left', fontsize=10)
ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', color='#e6e6e6', lw=0.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_1_correlation_heatmap.png'), dpi=170); plt.close(fig)
print('Saved to', OUT)

ctx = [c for c in CONTEXT if c in d.columns]
cc = d[kept + ctx].corr(method='spearman').loc[kept + ctx, ctx]
cc.round(3).to_csv(os.path.join(OUT, '3_context_correlation.csv'))
flag = [(a, b, cc.loc[a, b]) for a in kept + ctx for b in ctx if a != b and abs(cc.loc[a, b]) >= HIGH
        and not (a in ctx and ctx.index(a) > ctx.index(b))]
print('\nContext variables: pairs with |rho| >= 0.7 (social or context vs context):')
for a, b, r in sorted(flag, key=lambda x: -abs(x[2])):
    print(f"  {a:30s} {b:30s} {r:+.2f}")

### STEP 7a - Social groups: group postcodes by their census profile ONLY (no technical or storage variable).

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SOCIAL_DIR = os.path.dirname(HERE)
CODE = os.path.dirname(SOCIAL_DIR)
sys.path.insert(0, SOCIAL_DIR)

OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
VD = pd.read_csv(os.path.join(SOCIAL_DIR, '2_analysis_dataset', 'output', 'variable_dictionary.csv')).set_index('variable')['description'].to_dict()
POA_SHP = os.path.join(CODE, '2_load_data', 'data', 'POA_2021_AUST_GDA2020_SHP', 'POA_2021_AUST_GDA2020.shp')
PCA_VARIANCE = 0.80
K_RANGE = range(1, 11)
GROUP_COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']

d = pd.read_csv(os.path.join(SOCIAL_DIR, '2_analysis_dataset', 'output', 'analysis_dataset.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
V = pd.read_csv(os.path.join(SOCIAL_DIR, '3_collinearity', 'output', '3_kept_variables.csv'))['variable'].tolist()
X = ((d[V] - d[V].mean()) / d[V].std()).values
pca = PCA().fit(X)
nc = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), PCA_VARIANCE) + 1)
Zc = pca.transform(X)[:, :nc]
load = pd.DataFrame(pca.components_[:nc].T, index=V, columns=[f'PC{i + 1}' for i in range(nc)])
load.insert(0, 'description', [VD.get(v, v) for v in V])
load.loc['variance explained'] = ['', *pca.explained_variance_ratio_[:nc]]
load.to_csv(os.path.join(OUT, '7a_pca_loadings.csv'))

fits = {k: KMeans(k, n_init=50, random_state=1).fit(Zc) for k in K_RANGE}
ks = np.array(list(K_RANGE)); W = np.array([fits[k].inertia_ for k in ks])
xn = (ks - ks[0]) / (ks[-1] - ks[0]); yn = (W - W.min()) / (W.max() - W.min())
dist = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn + xn[-1] * yn[0] - yn[-1] * xn[0]) / np.hypot(yn[-1] - yn[0], xn[-1] - xn[0])
K = int(ks[np.argmax(dist)])
sel = pd.DataFrame({'k': ks, 'WCSS': W, 'pct_of_total_WCSS_removed_by_this_group': np.r_[np.nan, -np.diff(W) / W[0] * 100],
                    'distance_from_chord': dist, 'chosen': ks == K,
                    'sizes': ['/'.join(map(str, sorted(np.bincount(fits[k].labels_), reverse=True))) for k in ks]})
sel.round(3).to_csv(os.path.join(OUT, '7a_k_selection.csv'), index=False)


def short_name(z):
    if z['pct_age_20_29'] > 1 and z['pct_house'] < -1:
        return 'Young inner-urban'
    if z['pct_not_in_labour_force'] > 0.8:
        return 'Older, not in labour force'
    if z['pct_unemployed_15plus'] > 0.8:
        return 'Lower-income families'
    if z['log_income'] > 0.3:
        return 'Higher-income families'
    if z['pct_families_children'] > 0.3:
        return 'Suburban families'
    return 'Mixed'


def name_groups(labels):
    """Profiles (mean z) and names; groups numbered G1.. by size (largest first)."""
    prof = pd.DataFrame(X, index=d.index, columns=V).groupby(labels).mean()
    order = pd.Series(labels).value_counts().index
    short, long_, seen = {}, {}, {}
    for i, c in enumerate(order):
        z = prof.loc[c]
        s = short_name(z)
        seen[s] = seen.get(s, 0) + 1
        s = s if seen[s] == 1 else f'{s} {seen[s]}'
        hi = [VD.get(v, v).split(': ')[-1] for v in z.sort_values(ascending=False).index[:2]]
        lo = [VD.get(v, v).split(': ')[-1] for v in z.sort_values().index[:2]]
        short[c] = f'G{i + 1} {s}'
        long_[c] = f"G{i + 1} {s} (high {', '.join(hi)}; low {', '.join(lo)})"
    return prof, short, long_


lab = fits[K].labels_
prof, short, long_ = name_groups(lab)
g = pd.DataFrame({'group': pd.Series(lab, index=d.index).map(short), 'group_long': pd.Series(lab, index=d.index).map(long_)}, index=d.index)
g = g.join(pd.DataFrame(Zc, index=d.index, columns=[f'PC{i + 1}' for i in range(nc)]))
g.round(4).to_csv(os.path.join(OUT, '7a_postcode_groups.csv'), index_label='POSTCODE')

pr = prof.T
pr.columns = [short[c] for c in pr.columns]
pr = pr[sorted(pr.columns)]
pr.index = [VD.get(v, v) for v in pr.index]
pr.loc['postcodes'] = g['group'].value_counts().reindex(pr.columns).values
pr.round(3).to_csv(os.path.join(OUT, '7a_group_profiles.csv'))

pd.set_option('display.width', 250)
print('K SELECTION (elbow)'); print(sel.round(3).to_string(index=False)); print(f'chosen k = {K} ({nc} components, '
      f'{np.cumsum(pca.explained_variance_ratio_)[nc - 1]:.0%} of variance)')
print('\nGROUP PROFILES (mean z-score)'); print(pr.round(2).to_string())
print('\nRegion x group'); print(pd.crosstab(d['region'], g['group']).to_string())

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(ks, W, 'o-', color='#2a78d6')
ax.plot([ks[0], ks[-1]], [W[0], W[-1]], color='#bbbbbb', lw=0.8, ls=':')
ax.axvline(K, color='#c0392b', ls='--', lw=1)
ax.text(K + 0.15, W.max() * 0.9, f'elbow: k = {K}', color='#c0392b', fontsize=9)
ax.set_xticks(ks); ax.set_xlabel('Number of groups (k)'); ax.set_ylabel('Within-cluster sum of squares')
ax.set_title('Elbow method: choosing the number of social groups', loc='left', fontsize=10)
ax.grid(color='#eeeeee', lw=0.6); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_7a_1_elbow.png'), dpi=180); plt.close(fig)

names = sorted(g['group'].unique())
col = dict(zip(names, GROUP_COLOURS))
P = pr.drop('postcodes')
fig, ax = plt.subplots(figsize=(9.5, 7))
im = ax.imshow(P.values.astype(float), cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        v = float(P.iat[i, j])
        ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=8, color='white' if abs(v) > 1.2 else '#222222')
ax.set_yticks(range(P.shape[0]), [s.split(': ')[-1] for s in P.index], fontsize=8.5)
ax.set_xticks(range(P.shape[1]), [f"{n.replace(' ', chr(10), 1)}\n({int(pr.loc['postcodes', n])})" for n in P.columns], fontsize=8.5)
ax.xaxis.tick_top()
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label('Mean z-score (0 = average postcode)', fontsize=8.5)
ax.set_title(f"Social groups from the 7 kept census variables (k-means on {nc} principal components; k = {K} by the elbow method)",
             loc='left', fontsize=10, pad=55)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_7a_2_group_profiles.png'), dpi=180); plt.close(fig)

poa = gpd.read_file(POA_SHP, bbox=(151.8, -28.4, 153.7, -25.8)).rename(columns={'POA_CODE21': 'POSTCODE'})
gg = poa.merge(g[['group']].reset_index(), on='POSTCODE', how='inner')
fig, ax = plt.subplots(figsize=(7.5, 9))
poa[poa.intersects(gg.union_all().envelope)].plot(ax=ax, color='#f2f2f2', edgecolor='white', lw=0.2)
hand = []
for n in names:
    q = gg[gg['group'] == n]
    q.plot(ax=ax, color=col[n], edgecolor='white', lw=0.3)
    hand.append(plt.Rectangle((0, 0), 1, 1, color=col[n], label=f'{n} ({len(q)})'))
xmin, ymin, xmax, ymax = gg.total_bounds
ax.set_xlim(xmin - 0.05, xmax + 0.05); ax.set_ylim(ymin - 0.05, ymax + 0.05)
ax.legend(handles=hand, frameon=False, fontsize=8, loc='lower left')
ax.set_title('Social groups (census only); grey = not analysed', loc='left', fontsize=11); ax.set_axis_off()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_7a_3_group_map.png'), dpi=170); plt.close(fig)
print('Saved to', OUT)

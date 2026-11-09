### STEP 7c - Classification map: how does battery alignment vary across the social groups?

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import textwrap

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SOCIAL_DIR = os.path.dirname(HERE)
CODE = os.path.dirname(SOCIAL_DIR)
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
GROUPS = os.path.join(HERE, 'output', '7a_postcode_groups.csv')
CLASSES = os.path.join(HERE, 'output', '7b_alignment.csv')
FREQ_MAG = os.path.join(CODE, '2_load_data', '5_frequency_magnitude', 'output', 'frequency_magnitude_by_postcode.csv')
POA_SHP = os.path.join(CODE, '2_load_data', 'data', 'POA_2021_AUST_GDA2020_SHP', 'POA_2021_AUST_GDA2020.shp')
NPERM = 9999
CLASS_NAMES = {'A': 'Under-aligned (above capacity)', 'B': 'Under-aligned (below capacity)', 'C': 'Aligned', 'D': 'Over-aligned'}
CODE_OF = {'under-aligned (above capacity)': 'A', 'under-aligned (below capacity)': 'B', 'aligned': 'C', 'over-aligned': 'D'}

g = pd.read_csv(GROUPS, dtype={'POSTCODE': str}).set_index('POSTCODE')
c = pd.read_csv(CLASSES, dtype={'POSTCODE': str}).set_index('POSTCODE')
t = c.join(g[[col for col in g.columns if col.startswith('group')]], how='inner')
t['class'] = t['alignment'].map(CODE_OF)
t['classification'] = t['alignment'] + ' x ' + t['group']
t.round(4).to_csv(os.path.join(OUT, '7c_postcode_classification.csv'), index_label='POSTCODE')
groups = sorted(t['group'].unique())

def chi2(a, b, rng=np.random.default_rng(1)):
    """Chi-square of a x b with a permutation p-value (labels of b shuffled; contingency tables by bincount)."""
    obs = pd.crosstab(a, b)
    ai = pd.Categorical(a, categories=obs.index).codes
    bi = pd.Categorical(b, categories=obs.columns).codes
    r, k = obs.shape
    e = np.outer(obs.sum(1), obs.sum(0)) / obs.values.sum()
    stat = lambda o: ((o - e) ** 2 / e).sum()
    x = stat(obs.values)
    perm = np.array([stat(np.bincount(ai * k + rng.permutation(bi), minlength=r * k).reshape(r, k)) for _ in range(NPERM)])
    n = obs.values.sum()
    return obs, x, ((perm >= x - 1e-12).sum() + 1) / (NPERM + 1), np.sqrt(x / (n * (min(r, k) - 1)))


obs, X2, p, V = chi2(t['class'], t['group'])
n = obs.values.sum()
e = np.outer(obs.sum(1), obs.sum(0)) / n
rs, cs = obs.sum(1).values[:, None] / n, obs.sum(0).values[None, :] / n
res = pd.DataFrame((obs.values - e) / np.sqrt(e * (1 - rs) * (1 - cs)), index=obs.index, columns=obs.columns)
res.rename(index=CLASS_NAMES).round(2).to_csv(os.path.join(OUT, '7c_residuals.csv'))
ct = obs.copy(); ct['total'] = ct.sum(1); ct.loc['total'] = ct.sum()
ct.rename(index=CLASS_NAMES).to_csv(os.path.join(OUT, '7c_crosstab_postcodes.csv'))
col_share = obs / obs.sum(0)
row_share = obs.div(obs.sum(1), axis=0)

gap = t.groupby('group').agg(postcodes=('class', 'size'), solar_sites=('solar_sites', 'sum'), gap_MWh=('gap_MWh', 'sum'),
                             under_aligned=('class', lambda s: s.isin(['A', 'B']).sum()),
                             robust_under_aligned=('robustly_under_aligned', 'sum'),
                             median_gap_kWh=('gap_kWh_per_site', 'median'))
gap['under_aligned_above_capacity'] = t[t['class'] == 'A'].groupby('group').size().reindex(gap.index).fillna(0).astype(int)
gap['share_solar_sites'] = gap['solar_sites'] / gap['solar_sites'].sum()
gap['share_of_gap'] = gap['gap_MWh'] / gap['gap_MWh'].sum()
gap['gap_share_ratio'] = gap['share_of_gap'] / gap['share_solar_sites']
gap.round(3).to_csv(os.path.join(OUT, '7c_gap_by_group.csv'))

popc = t.pivot_table(index='class', columns='group', values='population', aggfunc='sum', fill_value=0)
popc.rename(index=CLASS_NAMES).to_csv(os.path.join(OUT, '7c_population_by_alignment.csv'))

hasneed = t['need_p90_kWh_per_site'] > 0
nd = t[hasneed & (t['group'].isin(t.loc[hasneed, 'group'].value_counts()[lambda x: x >= 5].index))]
_, X2n, pn, Vn = chi2(nd['class'], nd['group'])
summ = pd.DataFrame([
    {'test': 'alignment (4 categories) x group (main)', 'chi2': X2, 'p_perm': p, 'cramers_V': V},
    {'test': 'alignment x group, postcodes with P90 need > 0 (groups with >= 5 such postcodes)', 'chi2': X2n, 'p_perm': pn, 'cramers_V': Vn}])
summ.round(4).to_csv(os.path.join(OUT, '7c_summary.csv'), index=False)
pd.set_option('display.width', 250)
print('POSTCODES (alignment x group)'); print(ct.rename(index=CLASS_NAMES).to_string())
print('\nCOLUMN SHARES (each group across alignment)'); print(col_share.rename(index=CLASS_NAMES).round(2).to_string())
print('\nADJUSTED RESIDUALS'); print(res.rename(index=CLASS_NAMES).round(2).to_string())
print('\nGAP BY GROUP'); print(gap.round(3).to_string())
print('\nPEOPLE BY ALIGNMENT x GROUP'); print(popc.rename(index=CLASS_NAMES).round(0).astype(int).to_string())
print('\nTESTS'); print(summ.round(3).to_string(index=False))

poa = gpd.read_file(POA_SHP, bbox=(151.8, -28.4, 153.7, -25.8)).rename(columns={'POA_CODE21': 'POSTCODE'})
gg = poa.merge(t[['class', 'group']].reset_index(), on='POSTCODE', how='inner')
base = poa[poa.intersects(gg.union_all().envelope)]
xmin, ymin, xmax, ymax = gg.total_bounds

GROUP_COLOURS = ['#1f4e8c', '#0f6b4c', '#8a3519', '#6a3d9a', '#b0306a', '#4d4d4d']
gcol = dict(zip(groups, GROUP_COLOURS))
FADE = {'A': 0.0, 'B': 0.35, 'C': 0.62, 'D': 0.86}
ORDER_K = ['A', 'B', 'C', 'D']


def shade(hex_, k):
    rgb = np.array([int(hex_[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(rgb + (1 - rgb) * FADE[k])


def paint(ax_, data):
    for k in ORDER_K:
        for grp in groups:
            q = data[(data['class'] == k) & (data['group'] == grp)]
            if len(q):
                q.plot(ax=ax_, color=shade(gcol[grp], k), edgecolor='white', lw=0.3)


def class_group_matrix(ax_, values, fmt, title):
    """Rows = alignment (under-aligned dark -> no need pale), columns = social group; cells coloured like the map."""
    for i, k in enumerate(ORDER_K):
        for j, grp in enumerate(groups):
            ax_.add_patch(plt.Rectangle((j, -i), 0.96, 0.92, color=shade(gcol[grp], k)))
            ax_.text(j + 0.48, -i + 0.46, fmt(k, grp), ha='center', va='center', fontsize=9,
                     color='white' if k in ('A', 'B') else '#222222', fontweight='bold' if k == 'A' else 'normal')
    ax_.set_xlim(0, len(groups)); ax_.set_ylim(-len(ORDER_K) + 1, 1)
    ax_.set_yticks([-i + 0.46 for i in range(len(ORDER_K))], [CLASS_NAMES[k] for k in ORDER_K], fontsize=9)
    ax_.set_xticks([j + 0.48 for j in range(len(groups))], [textwrap.fill(g_, 16) + f'\n(n = {(t["group"] == g_).sum()})' for g_ in groups],
                   fontsize=8.5)
    ax_.xaxis.tick_top(); ax_.tick_params(length=0); ax_.spines[:].set_visible(False)
    ax_.set_title(title, loc='left', fontsize=10, pad=48)


npc = lambda k, grp: int(obs.at[k, grp]) if (k in obs.index and grp in obs.columns) else 0
gsize = t['group'].value_counts()

ZOOM = (152.94, -27.56, 153.11, -27.39)
fig = plt.figure(figsize=(17, 11))
ax = fig.add_axes([0.0, 0.0, 0.44, 0.92])
base.plot(ax=ax, color='#f2f2f2', edgecolor='white', lw=0.2)
paint(ax, gg)
ax.plot([ZOOM[0], ZOOM[2], ZOOM[2], ZOOM[0], ZOOM[0]], [ZOOM[1], ZOOM[1], ZOOM[3], ZOOM[3], ZOOM[1]], color='#222222', lw=1)
ax.set_xlim(xmin - 0.05, xmax + 0.05); ax.set_ylim(ymin - 0.05, ymax + 0.05); ax.set_axis_off()
ax.set_title(f'South East Queensland ({len(t)} postcodes)', fontsize=11)
zx = fig.add_axes([0.47, 0.50, 0.30, 0.42])
base.cx[ZOOM[0]:ZOOM[2], ZOOM[1]:ZOOM[3]].plot(ax=zx, color='#f2f2f2', edgecolor='white', lw=0.3)
gz = gg.cx[ZOOM[0]:ZOOM[2], ZOOM[1]:ZOOM[3]]
paint(zx, gz)
for _, r in gz.iterrows():
    pt = r.geometry.representative_point()
    if ZOOM[0] < pt.x < ZOOM[2] and ZOOM[1] < pt.y < ZOOM[3]:
        zx.text(pt.x, pt.y, r['POSTCODE'], fontsize=6.5, ha='center', va='center',
                color='white' if r['class'] in ('A', 'B') else '#333333')
zx.set_xlim(ZOOM[0], ZOOM[2]); zx.set_ylim(ZOOM[1], ZOOM[3]); zx.set_xticks([]); zx.set_yticks([])
for sp in zx.spines.values():
    sp.set_color('#222222')
zx.set_title('Inner Brisbane (box on the main map)', loc='left', fontsize=10)
from matplotlib.patches import ConnectionPatch
for (x0, y0), (x1, y1) in [((ZOOM[2], ZOOM[3]), (0, 1)), ((ZOOM[2], ZOOM[1]), (0, 0))]:
    fig.add_artist(ConnectionPatch(xyA=(x0, y0), coordsA=ax.transData, xyB=(x1, y1), coordsB=zx.transAxes, color='#999999', lw=0.7))
mx = fig.add_axes([0.62, 0.05, 0.36, 0.33])
class_group_matrix(mx, None, lambda k, grp: f'{npc(k, grp)} ({npc(k, grp) / gsize[grp]:.0%})' if npc(k, grp) else '0',
                   'Postcodes in each alignment and social group (% of the group)')
fig.suptitle('Battery alignment with P90 daily need by social group. Colour = social group; darkest = under-aligned (batteries < 0.5 x need) '
             'above hosting capacity, dark = under-aligned below capacity, light = aligned, palest = over-aligned (> 2 x or no need)', x=0.01, ha='left', fontsize=11)
fig.savefig(os.path.join(OUT, 'fig_7c_1_classification_map.png'), dpi=200, bbox_inches='tight'); plt.close(fig)

fig, px = plt.subplots(figsize=(9, 5))
P_ = popc.reindex(index=ORDER_K, columns=groups).fillna(0)
class_group_matrix(px, None, lambda k, grp: f'{P_.at[k, grp] / 1000:,.0f}k ({P_.at[k, grp] / P_[grp].sum():.0%})',
                   f'People in each alignment and social group, Census 2021 (% of the group; total {P_.values.sum() / 1e6:.2f} million)')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_7c_2_people.png'), dpi=200, bbox_inches='tight'); plt.close(fig)

fig, a2 = plt.subplots(figsize=(7.5, 3.6))
y = np.arange(len(gap))
a2.barh(y + 0.2, gap['share_solar_sites'], height=0.38, color='#bdbdbd', label='share of solar sites')
a2.barh(y - 0.2, gap['share_of_gap'], height=0.38, color=[gcol[g_] for g_ in gap.index], label='share of storage gap (P90 day)')
a2.set_yticks(y, [textwrap.fill(s_, 22) for s_ in gap.index], fontsize=8.5); a2.invert_yaxis()
a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
a2.set_title('Share of the storage gap vs share of solar sites, by social group', loc='left', fontsize=10)
a2.legend(frameon=False, fontsize=8); a2.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_7c_3_gap_share.png'), dpi=180); plt.close(fig)

print('Saved to', OUT)

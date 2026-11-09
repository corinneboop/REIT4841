### STEP 4 - Each selected variable on its own: which ones relate to the need for storage, and which to storage itself?

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
SOCIAL_DIR = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.join(SOCIAL_DIR, '2_analysis_dataset', 'output')
KEPT = os.path.join(SOCIAL_DIR, '3_collinearity', 'output', '3_kept_variables.csv')
d = pd.read_csv(os.path.join(DATA_DIR, 'analysis_dataset.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
vd = pd.read_csv(os.path.join(DATA_DIR, 'variable_dictionary.csv')).set_index('variable')
LABELS, THEME = vd['description'].to_dict(), vd['theme'].fillna('').to_dict()
CENSUS = pd.read_csv(KEPT)['variable'].tolist()
CONTEXT = ['business_share_pct', 'res_kWh_per_customer_day', 'log_bus_customers_per_1000_dw',
           'industrial_substation_pct', 'log_dwelling_density', 'res_tr_kVA_per_dwelling', 'log_ceg_per_solar_kVA',
           'pv_kVA_per_dwelling', 'avg_solar_kVA_per_site', 'major_city', 'dist_coast_km']
X_ALL = CENSUS + CONTEXT
NEED = {'p90': ('need_p90_kWh_per_site', 'P90 daily excess per solar site'),
        'annual': ('excess_kWh_per_site', 'Annual excess per solar site'),
        'days': ('pct_days_with_excess', '% of days with excess'),
        'p90cap': ('need_p90_weighted_kWh_per_site', 'Capacity-weighted P90 excess')}
STORE = {'kwh': ('battery_kWh_per_solar_site', 'Battery kWh per solar home'),
         'uptake': ('uptake_pct', 'Battery uptake (% of solar homes)'),
         'size': ('kWh_per_battery_site', 'kWh per battery')}
OUTCOMES = {**NEED, **STORE}
MAIN = ['p90', 'kwh']
ALPHA = 0.05


def strength(r):
    a = abs(r)
    return 'strong' if a >= 0.5 else 'moderate' if a >= 0.3 else 'weak'


t = pd.DataFrame({'block': ['census' if v in CENSUS else 'context' for v in X_ALL],
                  'theme': [THEME.get(v, '') for v in X_ALL], 'description': [LABELS[v] for v in X_ALL]}, index=X_ALL)
t.index.name = 'variable'
for k, (col, _) in OUTCOMES.items():
    res = [spearmanr(d[v], d[col], nan_policy='omit') for v in X_ALL]
    t[f'rho_{k}'] = [r[0] for r in res]
    t[f'p_{k}'] = [r[1] for r in res]
    t[f'q_{k}'] = multipletests(t[f'p_{k}'], method='fdr_bh')[1]
    t[f'strength_{k}'] = t[f'rho_{k}'].map(strength)
t.round(4).to_csv(os.path.join(OUT, '4_spearman.csv'))

pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 44)
for blk in ['census', 'context']:
    s = t[t['block'] == blk].sort_values('rho_p90', key=abs, ascending=False)
    show = pd.DataFrame({'description': s['description']})
    for k in OUTCOMES:
        show[k] = s[f'rho_{k}'].map('{:+.2f}'.format) + np.where(s[f'q_{k}'] < ALPHA, '*', ' ')
    print(f"\nSpearman rho, {blk} variables (* = BH q < 0.05), n = {len(d)}"); print(show.to_string())

ns = []
for kn, (cn, ln) in NEED.items():
    for ks, (cs, ls) in STORE.items():
        r, p = spearmanr(d[cn], d[cs], nan_policy='omit')
        ns.append({'need': ln, 'storage': ls, 'rho': r, 'p': p})
ns = pd.DataFrame(ns)
ns['q_BH'] = multipletests(ns['p'], method='fdr_bh')[1]
ns.round(4).to_csv(os.path.join(OUT, '4_need_vs_storage.csv'), index=False)
print('\nNEED vs STORAGE (Spearman rho; BH q)')
print(ns.assign(cell=ns['rho'].map('{:+.2f}'.format) + np.where(ns['q_BH'] < ALPHA, '*', ' '))
      .pivot(index='need', columns='storage', values='cell').to_string())
x, y = d['need_p90_kWh_per_site'], d['battery_kWh_per_solar_site']
print(f"Storage below P90 need in {int((y < x).sum())} of {len(d)} postcodes")

keys = list(OUTCOMES)
for_fig = pd.concat([t[t['block'] == 'census'].sort_values('rho_p90'), t[t['block'] == 'context'].sort_values('rho_p90')])
R = for_fig[[f'rho_{k}' for k in keys]].values
Q = for_fig[[f'q_{k}' for k in keys]].values
fig, ax = plt.subplots(figsize=(11, 0.42 * len(for_fig) + 2.4))
im = ax.imshow(R, cmap='RdBu_r', vmin=-0.75, vmax=0.75, aspect='auto')
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        ax.text(j, i, f"{R[i, j]:+.2f}" + ('*' if Q[i, j] < ALPHA else ''), ha='center', va='center', fontsize=7.5,
                color='white' if abs(R[i, j]) > 0.5 else '#222222', fontweight='bold' if keys[j] in MAIN else 'normal')
ax.set_xticks(range(len(keys)))
ax.set_xticklabels([('MAIN: ' if k in MAIN else '') + OUTCOMES[k][1] for k in keys], rotation=30, ha='right', fontsize=8.5)
ax.set_yticks(range(len(for_fig))); ax.set_yticklabels(for_fig['description'], fontsize=8.5)
nc = int((for_fig['block'] == 'census').sum())
ax.axhline(nc - 0.5, color='#222222', lw=1.2); ax.axvline(len(NEED) - 0.5, color='#222222', lw=1.2)
ax.text(len(NEED) / 2 - 0.5, -0.9, 'NEED (reverse energy)', ha='center', fontsize=9, fontweight='bold')
ax.text(len(NEED) + len(STORE) / 2 - 0.5, -0.9, 'STORAGE (batteries)', ha='center', fontsize=9, fontweight='bold')
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='Spearman rho')
ax.set_title(f'Each selected variable vs each outcome measure ({len(d)} Energex postcodes); * = BH q < 0.05. '
             'Above the line: census variables kept in step 3; below: context variables',
             loc='left', fontsize=10, pad=24)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_1_spearman_heatmap.png'), dpi=180); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={'width_ratios': [1, 1.15]})
ax = axes[0]
M = ns.pivot(index='need', columns='storage', values='rho').reindex(index=[l for _, l in NEED.values()], columns=[l for _, l in STORE.values()])
Mq = ns.pivot(index='need', columns='storage', values='q_BH').reindex(index=M.index, columns=M.columns)
im = ax.imshow(M.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M.values[i, j]:+.2f}" + ('*' if Mq.values[i, j] < ALPHA else ''), ha='center', va='center', fontsize=9)
ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=25, ha='right', fontsize=8.5)
ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=8.5)
ax.set_title('(a) Need vs storage measures: Spearman rho (* = BH q < 0.05)', loc='left', fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
ax = axes[1]
ax.scatter(x, y, s=22, color='#9ecae1', edgecolor='#2a78d6', lw=0.5, label='postcode')
top = max(3.2, y.max() * 1.1)
ax.plot([0, top], [0, top], color='#777777', lw=0.9, ls='--', label='storage = need')
ax.set_xlim(0, x.max() * 1.05); ax.set_ylim(0, top)
r0 = ns[(ns['need'] == NEED['p90'][1]) & (ns['storage'] == STORE['kwh'][1])].iloc[0]
ax.set_xlabel('P90 daily excess per solar site (kWh) - need'); ax.set_ylabel('Battery kWh per solar home - storage')
ax.set_title(f"(b) Main outcomes: rho = {r0['rho']:+.2f}; storage below need in {int((y < x).sum())} of {len(d)} postcodes",
             loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=8, loc='upper right')
ax.grid(color='#eeeeee', lw=0.5); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_2_need_vs_storage.png'), dpi=180); plt.close(fig)
print('Saved to', OUT)

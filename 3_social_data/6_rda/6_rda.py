### STEP 6 - Redundancy analysis (RDA): does a multivariate view corroborate the step 5 models?
### (Legendre & Legendre 2012, Numerical Ecology, ch. 11.)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SOCIAL_DIR = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.join(SOCIAL_DIR, '2_analysis_dataset', 'output')
KEPT = os.path.join(SOCIAL_DIR, '3_collinearity', 'output', '3_kept_variables.csv')
PERM = 999
rng = np.random.default_rng(1)

d = pd.read_csv(os.path.join(DATA_DIR, 'analysis_dataset.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
VD = pd.read_csv(os.path.join(DATA_DIR, 'variable_dictionary.csv')).set_index('variable')['description'].to_dict()
SOC = pd.read_csv(KEPT)['variable'].tolist()
OTHER = ['avg_solar_kVA_per_site', 'business_share_pct'] + [c for c in d.columns if c.startswith('region_')]
Yd = pd.DataFrame({'P90 daily excess (main)': np.log1p(d['need_p90_kWh_per_site']),
                   '% days with excess': d['pct_days_with_excess'],
                   'capacity-weighted P90': np.log1p(d['need_p90_weighted_kWh_per_site']),
                   'battery kWh per solar home (main)': np.log(d['battery_kWh_per_solar_site']),
                   'battery uptake': d['uptake_pct'],
                   'kWh per battery': np.log(d['kWh_per_battery_site'])}, index=d.index)
NEEDC, STOREC = list(Yd.columns[:3]), list(Yd.columns[3:])
ok = Yd.notna().all(1) & d[SOC + OTHER].notna().all(1)
Yd, d = Yd[ok], d[ok]
z = lambda df: (df - df.mean()) / df.std()
Y = z(Yd).values; Xs = z(d[SOC]).values; Xo = z(d[OTHER]).values
n, p = len(d), Xs.shape[1]
print(f'{n} postcodes, {Y.shape[1]} outcome measures, {p} census variables')


def fitted(Yv, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    B, *_ = np.linalg.lstsq(X1, Yv, rcond=None)
    return X1 @ B


def r2(Yv, X):
    Yh = fitted(Yv, X)
    return ((Yh - Yv.mean(0)) ** 2).sum() / ((Yv - Yv.mean(0)) ** 2).sum()


def adj(R2, k):
    return 1 - (1 - R2) * (n - 1) / (n - k - 1)


Yh = fitted(Y, Xs); R2 = r2(Y, Xs)
ev, U = np.linalg.eigh(np.cov(Yh.T)); o = np.argsort(ev)[::-1]; ev, U = ev[o], U[:, o]
k = int((ev > 1e-10).sum()); ev, U = ev[:k], U[:, :k]
total = np.trace(np.cov(Y.T))
lc = (Yh - Yh.mean(0)) @ U
cols = [f'RDA{a + 1}' for a in range(k)]
resp = pd.DataFrame([[np.corrcoef(Y[:, j], lc[:, a])[0, 1] for a in range(k)] for j in range(Y.shape[1])],
                    index=Yd.columns, columns=cols)
expl = pd.DataFrame([[np.corrcoef(Xs[:, j], lc[:, a])[0, 1] for a in range(k)] for j in range(p)],
                    index=[VD.get(v, v).split(': ')[-1].replace('household (log median)', 'household income (log)') for v in SOC], columns=cols)
F = (R2 / p) / ((1 - R2) / (n - p - 1))
Fp, evp = [], []
for _ in range(PERM):
    Yp = Y[rng.permutation(n)]
    R2p = r2(Yp, Xs); Fp.append((R2p / p) / ((1 - R2p) / (n - p - 1)))
    evp.append(np.sort(np.linalg.eigvalsh(np.cov(fitted(Yp, Xs).T)))[::-1][:k])
evp = np.array(evp)
axes = pd.DataFrame({'axis': cols, 'eigenvalue': ev, 'share_of_total_variance': ev / total,
                     'share_of_constrained_variance': ev / ev.sum(),
                     'p_perm': [(np.sum(evp[:, a] >= ev[a]) + 1) / (PERM + 1) for a in range(k)]})
resp_r2 = pd.DataFrame({'response': Yd.columns, 'R2_on_census': [r2(Y[:, [j]], Xs) for j in range(Y.shape[1])]})
resp_r2['adj_R2'] = adj(resp_r2['R2_on_census'], p)
vp = []
for block, cc in [('all six', list(Yd.columns)), ('need', NEEDC), ('storage', STOREC)]:
    Yb = z(Yd[cc]).values
    a_s = adj(r2(Yb, Xs), p); a_o = adj(r2(Yb, Xo), Xo.shape[1]); a_so = adj(r2(Yb, np.column_stack([Xs, Xo])), p + Xo.shape[1])
    vp.append({'response_block': block, 'census_adjR2': a_s, 'solar_load_location_adjR2': a_o, 'both_adjR2': a_so,
               '[a] census only': a_so - a_o, '[b] shared': a_s + a_o - a_so,
               '[c] solar, load & location only': a_so - a_s, '[d] unexplained': 1 - a_so})
vp = pd.DataFrame(vp)
summ = pd.Series({'n postcodes': n, 'outcome measures': Y.shape[1], 'census variables': p,
                  'constrained share (R2)': R2, 'adjusted R2': adj(R2, p), 'pseudo-F': F,
                  'p (999 permutations)': (np.sum(np.array(Fp) >= F) + 1) / (PERM + 1)})
summ.to_csv(os.path.join(OUT, '6_rda_summary.csv'), header=['value'])
axes.round(4).to_csv(os.path.join(OUT, '6_rda_axes.csv'), index=False)
resp.round(4).to_csv(os.path.join(OUT, '6_response_scores.csv'))
expl.round(4).to_csv(os.path.join(OUT, '6_explanatory_scores.csv'))
resp_r2.round(4).to_csv(os.path.join(OUT, '6_response_R2.csv'), index=False)
vp.round(4).to_csv(os.path.join(OUT, '6_variance_partitioning.csv'), index=False)
pd.set_option('display.width', 220)
print(summ.to_string())
print('\nAXES'); print(axes.round(3).to_string(index=False))
print('\nOUTCOMES: correlation with axes'); print(resp.iloc[:, :2].round(2).to_string())
print('\nCENSUS VARIABLES: correlation with axes'); print(expl.iloc[:, :2].round(2).to_string())
print('\nR2 OF EACH OUTCOME ON THE 7 CENSUS VARIABLES'); print(resp_r2.round(3).to_string(index=False))
print('\nVARIANCE PARTITIONING (adjusted R2)'); print(vp.round(3).to_string(index=False))

sites = pd.DataFrame(lc[:, :2], index=d.index, columns=['RDA1', 'RDA2'])
sc = sites / np.abs(sites).max().max()
fig, ax = plt.subplots(figsize=(11, 9))
ax.scatter(sc['RDA1'], sc['RDA2'], s=14, color='#bbbbbb', alpha=0.6, zorder=2, label=f'postcodes ({n})')
labs = []
for nm, rw in expl.iterrows():
    ax.annotate('', xy=(rw['RDA1'], rw['RDA2']), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#777777', lw=1.1), zorder=3)
    labs.append(dict(x=rw['RDA1'], y=rw['RDA2'], text=nm, color='#444444', weight='normal'))
for nm, rw in resp.iterrows():
    c = '#c0392b' if nm in NEEDC else '#1f3a5f'
    ax.annotate('', xy=(rw['RDA1'], rw['RDA2']), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=c, lw=2.4), zorder=4)
    labs.append(dict(x=rw['RDA1'], y=rw['RDA2'], text=nm, color=c, weight='bold'))
GAP = 0.075
for side in (-1, 1):
    L = sorted([l_ for l_ in labs if (l_['x'] < 0) == (side < 0)], key=lambda l_: l_['y'])
    ys = [l_['y'] for l_ in L]
    for _ in range(300):
        moved = False
        for k_ in range(1, len(ys)):
            if ys[k_] - ys[k_ - 1] < GAP:
                dd = (GAP - (ys[k_] - ys[k_ - 1])) / 2
                ys[k_ - 1] -= dd; ys[k_] += dd; moved = True
        if not moved:
            break
    for l_, yy in zip(L, ys):
        tx = l_['x'] + 0.04 * side
        ax.text(tx, yy, l_['text'], color=l_['color'], fontsize=10.5, fontweight=l_['weight'],
                ha='left' if side > 0 else 'right', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))
        if abs(yy - l_['y']) > 0.02:
            ax.plot([l_['x'], tx], [l_['y'], yy], color='#bbbbbb', lw=0.6, zorder=5)
ax.axhline(0, color='#cccccc', lw=0.7, zorder=1); ax.axvline(0, color='#cccccc', lw=0.7, zorder=1)
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.1, 1.1)
ax.set_xlabel(f"RDA1 ({axes.loc[0, 'share_of_constrained_variance']:.0%} of explained variance, p = {axes.loc[0, 'p_perm']:.3f})")
ax.set_ylabel(f"RDA2 ({axes.loc[1, 'share_of_constrained_variance']:.0%} of explained variance, p = {axes.loc[1, 'p_perm']:.3f})")
ax.set_title(f"RDA triplot: need (red), storage (navy), the 7 kept census variables (grey)\n"
             f"census variables explain {R2:.0%} (adjusted {adj(R2, p):.0%}) of the six outcome measures", loc='left', fontsize=11)
ax.legend(frameon=False, fontsize=9, loc='lower left')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_1_triplot.png'), dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 3.6))
parts = ['[a] census only', '[b] shared', '[c] solar, load & location only', '[d] unexplained']
pcol = ['#7570b3', '#bcbddc', '#e08214', '#eeeeee']
yv = np.arange(len(vp))[::-1]; left = np.zeros(len(vp))
for pt, c in zip(parts, pcol):
    val = vp[pt].clip(lower=0).values
    ax.barh(yv, val, left=left, color=c, edgecolor='white', label=pt)
    for yy, l, v in zip(yv, left, val):
        if v > 0.04:
            ax.text(l + v / 2, yy, f'{v:.0%}', ha='center', va='center', fontsize=8)
    left += val
ax.set_yticks(yv); ax.set_yticklabels(vp['response_block'])
ax.set_xlabel('Share of variance (adjusted R2; negative fractions shown as 0)')
ax.set_title('Variance partitioning: 7 census variables vs solar size, business share and region', loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.28))
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_2_variance_partitioning.png'), dpi=170); plt.close(fig)
print('Saved to', OUT)

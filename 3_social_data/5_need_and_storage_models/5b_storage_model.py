### STEP 5b - Which variables explain how much battery STORAGE postcodes have? Multiple regression with the
### capital framing of Best et al. (2021), "Actual uptake of home batteries: the key roles of capital and policy",
### Energy Policy 151, 112186.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_reset

HERE = os.path.dirname(os.path.abspath(__file__))
SOCIAL_DIR = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.join(SOCIAL_DIR, '2_analysis_dataset', 'output')
KEPT = os.path.join(SOCIAL_DIR, '3_collinearity', 'output', '3_kept_variables.csv')
ALPHA = 0.05
K_NEIGHBOURS = 6
REF_REGION = 'Brisbane'
CLUSTER = 'main_substation'
Y = 'log_battery_kWh_per_solar_site'
CAPITAL = ['avg_solar_kVA_per_site', 'log_income']
SOCIAL = [v for v in pd.read_csv(KEPT)['variable'] if v not in CAPITAL]
LOAD = ['business_share_pct']
rng = np.random.default_rng(1)

d = pd.read_csv(os.path.join(DATA_DIR, 'analysis_dataset.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
VD = pd.read_csv(os.path.join(DATA_DIR, 'variable_dictionary.csv')).set_index('variable')['description'].to_dict()
LABELS = {**VD, 'avg_solar_kVA_per_site': 'Capital: solar system size (kVA per site)',
          'log_income': 'Capital: household income (log median)'}
REG = [c for c in d.columns if c.startswith('region_')]
for c in REG:
    LABELS[c] = f"Region: {c[len('region_'):]} (vs {REF_REGION})"
d = d.dropna(subset=[Y] + CAPITAL + SOCIAL + LOAD)
d['log_kWh_per_battery'] = np.log(d['kWh_per_battery_site'])
MAIN = CAPITAL + SOCIAL + LOAD + REG
BLOCKS = [('Main model', MAIN)]
print(f"{len(d)} Energex postcodes, battery kWh per solar home median {d['battery_kWh_per_solar_site'].median():.2f} "
      f"(range {d['battery_kWh_per_solar_site'].min():.2f}-{d['battery_kWh_per_solar_site'].max():.2f})")


def design(data, cols):
    X = data[cols].copy()
    cont = [c for c in cols if not c.startswith('region_')]
    X[cont] = (X[cont] - data[cont].mean()) / data[cont].std()
    return sm.add_constant(X)


def cluster_cov(data):
    return dict(cov_type='cluster', cov_kwds={'groups': data[CLUSTER].astype('category').cat.codes.values})


def vifs(data, cols):
    X = sm.add_constant(data[cols])
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=cols)


def morans_i(data, resid):
    xy = data[['lat', 'long']].values
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(dist, np.inf)
    W = np.zeros_like(dist); nn = np.argsort(dist, axis=1)[:, :K_NEIGHBOURS]
    W[np.arange(len(xy))[:, None], nn] = 1 / K_NEIGHBOURS
    zr = np.asarray(resid) - np.mean(resid)
    I = zr @ W @ zr / (zr @ zr)
    perm = [(p @ W @ p) / (p @ p) for p in (rng.permutation(zr) for _ in range(999))]
    return I, (np.sum(np.array(perm) >= I) + 1) / 1000


def stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''


def pct(b):
    return 100 * (np.exp(b) - 1)


rows, fits = [], []
for bname, cols in BLOCKS:
    X = design(d, cols)
    p = sm.OLS(d[Y], X).fit()
    c = sm.OLS(d[Y], X).fit(**cluster_cov(d))
    ci = c.conf_int()
    for t in cols:
        rows.append({'model': bname, 'variable': t, 'description': LABELS.get(t, t), 'coef_per_SD': c.params[t],
                     'pct_per_SD': pct(c.params[t]), 'se': c.bse[t], 'ci_low': ci.loc[t, 0], 'ci_high': ci.loc[t, 1],
                     'p': c.pvalues[t], 'sig': stars(c.pvalues[t])})
    fits.append({'model': bname, 'n': int(p.nobs), 'k': len(cols), 'R2': p.rsquared, 'adj_R2': p.rsquared_adj,
                 'AIC': p.aic, 'max_VIF': vifs(d, cols).max()})
I, Ip = morans_i(d, p.resid)
diag = pd.DataFrame([{'model': BLOCKS[-1][0],
                      'RESET_p': float(linear_reset(p, power=3, use_f=True).pvalue),
                      'Morans_I': I, 'Morans_I_p': Ip, 'max_VIF': vifs(d, MAIN).max()}])
st, fits = pd.DataFrame(rows), pd.DataFrame(fits)
st.round(4).to_csv(os.path.join(OUT, '5b_model.csv'), index=False)
fits.round(4).to_csv(os.path.join(OUT, '5b_model_fit.csv'), index=False)
diag.round(4).to_csv(os.path.join(OUT, '5b_diagnostics.csv'), index=False)

pd.set_option('display.width', 230)
w = st.assign(cell=st['pct_per_SD'].map('{:+.1f}%'.format) + st['sig'] + ' (' + st['coef_per_SD'].map('{:+.3f}'.format) + ')'
              ).pivot(index='variable', columns='model', values='cell')
w = w.reindex(MAIN)[[b for b, _ in BLOCKS]].fillna('')
w.index = [LABELS.get(v, v) for v in w.index]
for k in ['R2', 'adj_R2', 'max_VIF']:
    w.loc[k] = fits.set_index('model')[k].round(3).astype(str).reindex(w.columns).values
print('\nBattery kWh per solar home: % change per +1 SD (log coefficient); *** p<0.01 ** p<0.05 * p<0.1; SE clustered')
print(w.to_string())
print('\nDIAGNOSTICS (main model)'); print(diag.round(3).to_string(index=False))

NEED = 'log_need_p90_per_site'
un = sm.OLS(d[Y], design(d, [NEED])).fit(**cluster_cov(d))
ad = sm.OLS(d[Y], design(d, MAIN + [NEED])).fit(**cluster_cov(d))
need_added = pd.DataFrame([{'need_measure': 'log(1 + P90 daily excess per solar site)', 'n': int(ad.nobs),
                            'unadjusted_pct_per_SD': pct(un.params[NEED]), 'unadjusted_p': un.pvalues[NEED],
                            'adjusted_pct_per_SD': pct(ad.params[NEED]), 'adjusted_p': ad.pvalues[NEED],
                            'R2_main': p.rsquared, 'R2_main_plus_need': ad.rsquared}])
need_added.round(4).to_csv(os.path.join(OUT, '5b_need_added.csv'), index=False)
print('\nDOES STORAGE FOLLOW NEED? (% change in battery kWh per solar home per +1 SD of P90 need)')
print(need_added.round(3).to_string(index=False))

vars_ = CAPITAL + SOCIAL + LOAD

m = st[st['model'] == 'Main model'].set_index('variable')
fig, ax = plt.subplots(figsize=(9.5, 6.5))
for i, v in enumerate(vars_[::-1]):
    ax.hlines(i, pct(m.loc[v, 'ci_low']), pct(m.loc[v, 'ci_high']), color='#08306b', lw=1.6)
    ax.scatter(m.loc[v, 'pct_per_SD'], i, s=36, zorder=3, edgecolor='#08306b',
               facecolor='#08306b' if m.loc[v, 'p'] < ALPHA else 'white')
ax.axvline(0, color='#555555', lw=0.8)
ax.set_yticks(range(len(vars_))); ax.set_yticklabels([LABELS[v] for v in vars_[::-1]], fontsize=8.5)
ax.set_xlabel('% change in the outcome per +1 SD of the variable (95% CI, clustered; filled = p < 0.05)')
ax.set_title('Battery storage: main model coefficients', loc='left', fontsize=10)
ax.grid(axis='x', color='#e6e6e6', lw=0.6); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_5b_1_coefficients.png'), dpi=180); plt.close(fig)
print('Saved to', OUT)

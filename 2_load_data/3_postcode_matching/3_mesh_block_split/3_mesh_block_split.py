### STEP 3.3 - Mesh Block methods: split each zone substation's load between the postcodes it supplies through its
### distribution transformers. Each transformer stands in one ABS Mesh Block (step 3.2), and Mesh Blocks nest exactly
### in postcodes, so each transformer belongs to one postcode. Three weightings: equal, rated kVA and kVA x utilisation.

import os
import re
import itertools
import sys
import numpy as np
import pandas as pd

MIN_COVERAGE = 0.9
K_FOLDS = 10
SEED = 1
TYPES = ['POLE', 'PADMOUNT', 'GROUND', 'C&I']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MATCH_DIR = os.path.dirname(SCRIPT_DIR)
NET = os.path.join(MATCH_DIR, '2_network_matching', 'output', 'network')
SOLAR_FILE = os.path.join(NET, 'site_postcode_solar.csv')
sys.path.insert(0, MATCH_DIR)
sys.dont_write_bytecode = True
from split_tools import solar_shares, BUSINESS, BUSINESS_TYPES
SITE_LOAD_FILE = os.path.join(os.path.dirname(MATCH_DIR), '1_join_substation_data', 'output', 'load_MW_by_site.csv')
OUT = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT, exist_ok=True)


def nnls(A, b):
    """Non-negative least squares by trying every set of free variables (only 4 here, so 15 sets)."""
    best, n = None, A.shape[1]
    for k in range(1, n + 1):
        for cols in itertools.combinations(range(n), k):
            x_sub = np.linalg.lstsq(A[:, cols], b, rcond=None)[0]
            if (x_sub < 0).any():
                continue
            x = np.zeros(n); x[list(cols)] = x_sub
            r = np.sum((A @ x - b) ** 2)
            if best is None or r < best[1]:
                best = (x, r)
    return best[0]


site_load = pd.read_csv(SITE_LOAD_FILE, index_col='DATETIME', parse_dates=True)
sites = site_load.columns.tolist()
t = pd.read_csv(os.path.join(NET, 'transformers.csv'), dtype={'POSTCODE': str, 'MB_CODE': str})
t = t[t['SITE'].isin(sites)].copy()
t['LAND_USE'] = np.where(t['MB_CATEGORY'].isin(BUSINESS) | t['TR_TYPE'].isin(BUSINESS_TYPES), 'business',
                         np.where(t['MB_CATEGORY'] == 'Residential', 'residential', 'other'))
summary = pd.read_csv(os.path.join(NET, 'site_summary.csv')).set_index('SITE')
link = pd.read_csv(os.path.join(NET, 'meshblocks.csv'), dtype={'POSTCODE': str})
link = link[link['SITE'].isin(sites)]
dw_given = link.groupby(['SITE', 'POSTCODE'])['DWELLINGS_GIVEN'].sum()
suburbs = (link[link['DWELLINGS_GIVEN'] > 0.5].groupby(['SITE', 'POSTCODE'])['SAL_NAME']
           .apply(lambda x: ', '.join(sorted(set(re.sub(r'\s*\(.*\)$', '', str(v)) for v in x)))))
print(f"{len(sites)} sites, {len(t):,} transformers on their feeders ({t['W'].sum() / 1000:,.0f} MVA); "
      f"{t['POSTCODE'].isna().sum()} without a postcode are left out of the split")
print('Transformer labels (count):', t['LAND_USE'].value_counts().to_dict())

coverage = site_load.notna().mean()
avg_mw = site_load.mean()
fit_sites = coverage.index[coverage >= MIN_COVERAGE]
mva = t.pivot_table(index='SITE', columns='TR_TYPE', values='W', aggfunc='sum').reindex(
    index=sites, columns=TYPES).fillna(0) / 1000
A, b = mva.loc[fit_sites].values, avg_mw[fit_sites].values
u = pd.Series(nnls(A, b), index=TYPES)
c_one = nnls(A.sum(axis=1, keepdims=True), b)[0]

rng = np.random.default_rng(SEED)
fold = pd.Series(rng.permutation(len(fit_sites)) % K_FOLDS, index=fit_sites)
pred_cv = pd.DataFrame(index=fit_sites, columns=['mb_kva', 'mb_kva_utilisation'], dtype=float)
u_folds = []
for k in range(K_FOLDS):
    tr, te = fold.index[fold != k], fold.index[fold == k]
    uk = nnls(mva.loc[tr].values, avg_mw[tr].values)
    ck = nnls(mva.loc[tr].values.sum(axis=1, keepdims=True), avg_mw[tr].values)[0]
    u_folds.append(uk)
    pred_cv.loc[te, 'mb_kva_utilisation'] = mva.loc[te].values @ uk
    pred_cv.loc[te, 'mb_kva'] = mva.loc[te].values.sum(axis=1) * ck
u_folds = pd.DataFrame(u_folds, columns=TYPES)

tab = pd.DataFrame({'U_MW_PER_MVA': u, 'U_CV_MIN': u_folds.min(), 'U_CV_MAX': u_folds.max(),
                    'TRANSFORMERS': t.groupby('TR_TYPE').size().reindex(TYPES),
                    'MEDIAN_KVA': t.groupby('TR_TYPE')['W'].median().reindex(TYPES),
                    'MVA': mva.sum(), 'SHARE_OF_KVA': mva.sum() / mva.sum().sum()})
tab['SHARE_OF_FITTED_LOAD'] = tab['U_MW_PER_MVA'] * tab['MVA'] / (tab['U_MW_PER_MVA'] * tab['MVA']).sum()
tab.loc['ALL (mb_kva)'] = [c_one, np.nan, np.nan, len(t), t['W'].median(), mva.values.sum(), 1, 1]
tab.round(4).to_csv(os.path.join(OUT, 'utilisation_factors.csv'), index_label='TR_TYPE')

fit = pd.DataFrame({'COVERAGE': coverage, 'METERED_AVG_MW': avg_mw, 'MVA': mva.sum(axis=1),
                    'PRED_mb_kva': mva.sum(axis=1) * c_one, 'PRED_mb_kva_utilisation': mva @ u,
                    'CV_PRED_mb_kva': pred_cv['mb_kva'], 'CV_PRED_mb_kva_utilisation': pred_cv['mb_kva_utilisation'],
                    'LOAD_CATEGORY': summary['LOAD_CATEGORY'].reindex(sites)})
fit.round(4).to_csv(os.path.join(OUT, 'site_fit.csv'), index_label='SITE')


def site_errors(col):
    f = fit.loc[fit_sites]
    e = f[col] / f['METERED_AVG_MW'] - 1
    r2 = 1 - ((f[col] - f['METERED_AVG_MW']) ** 2).sum() / ((f['METERED_AVG_MW'] - f['METERED_AVG_MW'].mean()) ** 2).sum()
    return 100 * e.abs().median(), r2


print(f"\nFit of u on {len(fit_sites)} sites (average load over the year, MW per MVA of rated capacity):")
print(tab.round(3).to_string())
for m in ['mb_kva', 'mb_kva_utilisation']:
    e_in, r2_in = site_errors(f'PRED_{m}')
    e_cv, r2_cv = site_errors(f'CV_PRED_{m}')
    print(f"  {m:16s} site average load: median error {e_in:4.1f}% (R2 {r2_in:.3f}); "
          f"cross-validated {e_cv:4.1f}% (R2 {r2_cv:.3f})")

weights = {'mb_equal': pd.Series(1.0, index=t.index), 'mb_kva': t['W'],
           'mb_kva_utilisation': t['W'] * t['TR_TYPE'].map(u).fillna(0)}
for name, w in weights.items():
    d = t.assign(WEIGHT=w, KVA=t['W']).dropna(subset=['POSTCODE'])
    d['BUS_W'] = d['WEIGHT'].where(d['LAND_USE'] == 'business', 0)
    d['OTH_W'] = d['WEIGHT'].where(d['LAND_USE'] == 'other', 0)
    m = d.groupby(['SITE', 'POSTCODE']).agg(TRANSFORMERS=('KVA', 'size'), KVA=('KVA', 'sum'),
                                            WEIGHT=('WEIGHT', 'sum'), BUS_W=('BUS_W', 'sum'),
                                            OTH_W=('OTH_W', 'sum')).reset_index()
    m = m[m['WEIGHT'] > 0]
    m['SHARE'] = m['WEIGHT'] / m.groupby('SITE')['WEIGHT'].transform('sum')
    m['POSTCODE_FRACTION'] = m['WEIGHT'] / m.groupby('POSTCODE')['WEIGHT'].transform('sum')
    m['BUSINESS_FRACTION'] = m['BUS_W'] / m['WEIGHT']
    m['OTHER_FRACTION'] = m['OTH_W'] / m['WEIGHT']
    key = list(zip(m['SITE'], m['POSTCODE']))
    m['DWELLINGS_GIVEN'] = [dw_given.get(k, 0) for k in key]
    m['SUBURBS'] = [suburbs.get(k, '') for k in key]
    m['BUSES'] = m['SITE'].map(summary['BUSES'])
    m['LOAD_CATEGORY'] = m['SITE'].map(summary['LOAD_CATEGORY'])
    m['SITES_IN_POSTCODE'] = m.groupby('POSTCODE')['SITE'].transform('nunique')
    m['SOLAR_SHARE'] = solar_shares(m, SOLAR_FILE)
    assert ((m.groupby('SITE')['SHARE'].sum() - 1).abs() < 1e-9).all()
    assert ((m.groupby('SITE')['SOLAR_SHARE'].sum() - 1).abs() < 1e-9).all()
    cols = ['SITE', 'BUSES', 'LOAD_CATEGORY', 'POSTCODE', 'SHARE', 'SOLAR_SHARE', 'POSTCODE_FRACTION', 'TRANSFORMERS', 'KVA',
            'WEIGHT', 'BUSINESS_FRACTION', 'OTHER_FRACTION', 'DWELLINGS_GIVEN', 'SUBURBS', 'SITES_IN_POSTCODE']
    folder = os.path.join(OUT, name)
    os.makedirs(folder, exist_ok=True)
    m[cols].round(6).to_csv(os.path.join(folder, 'site_postcode_mapping.csv'), index=False)

    print(f"{name:16s} {m['SITE'].nunique()} sites -> {m['POSTCODE'].nunique()} postcodes, {len(m)} site-postcode pairs")
print('Saved to', OUT)

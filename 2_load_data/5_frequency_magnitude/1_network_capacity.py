### Step 5.1 - Postcode reverse power flow vs the network's hosting capacity (DAPR Capacity for Embedded Generation).
### Run 4_reverse_flow first. 2_frequency_magnitude.py (same folder) uses the daily % of CEG calculated here.

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD_DIR = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
SITE_LOAD = os.path.join(LOAD_DIR, '1_join_substation_data', 'output', 'load_MW_by_site.csv')
MAPPING = os.path.join(LOAD_DIR, '3_postcode_matching', '5_compare_methods', 'output', 'selected', 'site_postcode_mapping.csv')
DAPR = os.path.join(LOAD_DIR, '2_dapr_parameters', 'output', 'dapr_parameters_by_bus.csv')
DAPR_SITE = os.path.join(LOAD_DIR, '2_dapr_parameters', 'output', 'dapr_validation.csv')
SOLAR = os.path.join(LOAD_DIR, '3_postcode_matching', '2_network_matching', 'output', 'network', 'site_postcode_solar.csv')
STEP4 = os.path.join(LOAD_DIR, '4_reverse_flow', 'output')
START, END = '2025-07-01', '2026-06-30'
UPSAMPLE = 6
MIN_DAYS_ABOVE = 5
MIN_REVERSE_DAYS = 10
NEAR_PCT = 50
ABOVE_SHARE_PCT = 5
NO_CAP_PCT = 50
MIN_SITE_FRACTION = 0.05
CEG = 'CAPACITY_FOR_EMBEDDED_GENERATION_MVA__'
N1 = 'N_1_CAPACITY_FOR_EMBEDDED_GENERATION_MVA__'
KINDS = {'main': ('2025/26 CEG (main)', CEG + 'S2526', CEG + 'W2025', CEG + 'W2026'),
         'n1': ('N-1 CEG (one transformer out)', N1 + 'S2526', N1 + 'W2025', N1 + 'W2026')}
SUFFIX = {'main': '', 'n1': '_N1'}
DT = 0.5 / UPSAMPLE

period = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta('23h30min'), freq='30min')
days = pd.DatetimeIndex(sorted(set(period.normalize())))
D = len(days)
site = pd.read_csv(SITE_LOAD, index_col='DATETIME', parse_dates=True).reindex(period)
mp = pd.read_csv(MAPPING, dtype={'POSTCODE': str})
cap4 = pd.read_csv(os.path.join(STEP4, 'postcode_pv_capacity.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
postcodes = cap4.index.tolist()
SS = mp.pivot_table(index='SITE', columns='POSTCODE', values='SOLAR_SHARE', aggfunc='sum').reindex(
    columns=postcodes).fillna(0)
SS = SS[(SS.sum(axis=1) > 0) & SS.index.isin(site.columns)]
sites = SS.index.tolist()
S, P = len(sites), len(postcodes)
R_site = (-site[sites].clip(upper=0)).fillna(0).values.reshape(D, 48, S).astype(np.float32)

daily4 = pd.read_csv(os.path.join(STEP4, 'reverse_flow_daily_by_postcode.csv'), dtype={'POSTCODE': str},
                     parse_dates=['DATE'])
ok = ~daily4.pivot(index='DATE', columns='POSTCODE', values='DAY_FLAGGED').reindex(index=days, columns=postcodes) \
    .fillna(True).astype(bool).values

dapr = pd.read_csv(DAPR).merge(pd.read_csv(DAPR_SITE)[['CODE', 'SITE']], on='CODE')
num = lambda s: pd.to_numeric(s, errors='coerce').sum(min_count=1)
warm = np.asarray(days.month.isin([9, 10, 11, 12, 1, 2]))
y2025 = np.asarray(days.year == 2025)
CS = {}
for k, (_, cs, cw1, cw2) in KINDS.items():
    v = {c: dapr.groupby('SITE')[c].apply(num).reindex(sites).values for c in (cs, cw1, cw2)}
    CS[k] = np.where(warm[:, None], v[cs], np.where(y2025[:, None], v[cw1], v[cw2]))


def upsample(x):
    """(D, 48, n) half-hourly -> (D, 283, n) at 5 min, linear within each day."""
    f = np.arange((48 - 1) * UPSAMPLE + 1)
    i = np.minimum(f // UPSAMPLE, 46)
    w = ((f - i * UPSAMPLE) / UPSAMPLE).astype(np.float32)
    return x[:, i, :] * (1 - w)[None, :, None] + x[:, i + 1, :] * w[None, :, None]


SSv = SS.values.astype(np.float32)
fr = pd.read_csv(SOLAR, dtype={'POSTCODE': str}).pivot_table(index='SITE', columns='POSTCODE', values='DWELLING_FRACTION',
                                                             aggfunc='sum').reindex(index=sites, columns=postcodes).fillna(0)
material = ((fr.values >= MIN_SITE_FRACTION) & (SSv > 0)).astype(np.float32)
Rf_site = upsample(R_site)
E_site = Rf_site.sum(axis=1) * DT
peak_site = Rf_site.max(axis=1)
Rf_all = upsample(np.einsum('dts,sp->dtp', R_site, SSv))
daily = {'REVERSE_MWh': Rf_all.sum(axis=1) * DT, 'PEAK_REVERSE_MW': Rf_all.max(axis=1)}
del Rf_all

for k in KINDS:
    has = np.nan_to_num(CS[k]) > 0
    Cs = np.where(has, CS[k], np.inf).astype(np.float32)
    A = np.clip(Rf_site - Cs[:, None, :], 0, None)
    Eh = E_site * has
    daily[f'CEG_MW_{k}'] = np.where(has, CS[k], 0).astype(np.float32) @ SSv
    daily[f'REVERSE_WITH_CEG_MWh_{k}'] = Eh @ SSv
    with np.errstate(divide='ignore', invalid='ignore'):
        daily[f'PEAK_PCT_OF_CEG_{k}'] = 100 * ((Eh * np.where(has, peak_site / Cs, 0)) @ SSv) / (Eh @ SSv)
    daily[f'ABOVE_MWh_{k}'] = (A.sum(axis=1) * DT) @ SSv
    on = ((A > 0).reshape(-1, S).astype(np.float32) @ material).reshape(D, -1, P) > 0
    daily[f'HOURS_ABOVE_{k}'] = on.sum(axis=1) * DT
    del A, on
daily['WITHIN_MWh_main'] = daily['REVERSE_WITH_CEG_MWh_main'] - daily['ABOVE_MWh_main']
for k, v in daily.items():
    daily[k] = np.where(ok, v, np.nan)

dd = pd.concat({k: pd.DataFrame(v, index=days, columns=postcodes).stack(future_stack=True)
                for k, v in daily.items()}, axis=1)
dd.index.names = ['DATE', 'POSTCODE']
dd = dd.dropna(subset=['REVERSE_MWh'])
dd.round(4).to_csv(os.path.join(OUT, 'capacity_daily_by_postcode.csv'))

g = dd.groupby('POSTCODE')
rv = dd[dd['REVERSE_MWh'] > 0].groupby('POSTCODE')
month = dd.index.get_level_values('DATE').month
sm, wn = dd[month.isin([9, 10, 11, 12, 1, 2])].groupby('POSTCODE'), dd[month.isin([3, 4, 5, 6, 7, 8])].groupby('POSTCODE')
p = pd.DataFrame(index=pd.Index(postcodes, name='POSTCODE'))
p['DAYS_USED'] = g.size()
scale = 365 / p['DAYS_USED']
p['DAYS_WITH_REVERSE'] = rv.size().reindex(p.index).fillna(0).astype(int)
p['ANNUAL_REVERSE_MWh'] = g['REVERSE_MWh'].sum() * scale
p['NO_CAPACITY_PCT'] = 100 * (1 - g['REVERSE_WITH_CEG_MWh_main'].sum() / g['REVERSE_MWh'].sum().replace(0, np.nan))
p['CEG_SUMMER_MW'] = sm['CEG_MW_main'].median()
p['CEG_WINTER_MW'] = wn['CEG_MW_main'].median()
p['CEG_N1_SUMMER_MW'] = sm['CEG_MW_n1'].median()
p['MEDIAN_PEAK_MW'] = rv['PEAK_REVERSE_MW'].median()
p['P90_PEAK_MW'] = rv['PEAK_REVERSE_MW'].quantile(0.9)
p['MAX_PEAK_MW'] = g['PEAK_REVERSE_MW'].max()
for k, sfx in SUFFIX.items():
    p[f'P90_PEAK_PCT_OF_CEG{sfx}'] = rv[f'PEAK_PCT_OF_CEG_{k}'].quantile(0.9)
    p[f'DAYS_ABOVE_{k}'] = dd[dd[f'HOURS_ABOVE_{k}'] > 0].groupby('POSTCODE').size().reindex(p.index).fillna(0).astype(int)
    p[f'HOURS_ABOVE_{k}'] = g[f'HOURS_ABOVE_{k}'].sum() * scale
    p[f'ANNUAL_ABOVE_MWh_{k}_RAW'] = g[f'ABOVE_MWh_{k}'].sum() * scale
    p[f'ANNUAL_ABOVE_MWh_{k}'] = p[f'ANNUAL_ABOVE_MWh_{k}_RAW'].where(p[f'DAYS_ABOVE_{k}'] >= MIN_DAYS_ABOVE, 0)
p['ANNUAL_WITHIN_MWh'] = g['REVERSE_WITH_CEG_MWh_main'].sum() * scale - p['ANNUAL_ABOVE_MWh_main']
p['PCT_ABOVE_CEG'] = np.where(p['ANNUAL_REVERSE_MWh'] > 0, 100 * p['ANNUAL_ABOVE_MWh_main'] / p['ANNUAL_REVERSE_MWh'], np.nan)

contrib = SS.mul(pd.Series(R_site.sum(axis=(0, 1)), index=sites), axis=0)
del R_site, Rf_site
p['MAIN_REVERSE_SITE'] = contrib.idxmax().where(contrib.max() > 0)
p['MAIN_SITE_SHARE_PCT'] = 100 * contrib.max() / contrib.sum().replace(0, np.nan)
p['SITES_SUPPLYING'] = (SS > 0).sum()

p = p.join(cap4[['SOLAR_kVA', 'SOLAR_SITES']])
for c, n in [('ANNUAL_REVERSE_MWh', 'REVERSE'), ('ANNUAL_WITHIN_MWh', 'WITHIN'), ('ANNUAL_ABOVE_MWh_main', 'ABOVE'),
             ('ANNUAL_ABOVE_MWh_n1', 'ABOVE_N1')]:
    p[f'{n}_kWh_PER_SOLAR_SITE'] = 1000 * p[c] / p['SOLAR_SITES']
CEG_COL = {'main': 'CEG_SUMMER_MW', 'n1': 'CEG_N1_SUMMER_MW'}


def status_for(k='main', min_days=MIN_DAYS_ABOVE, near_pct=NEAR_PCT):
    return pd.Series(np.select(
        [p['DAYS_WITH_REVERSE'] == 0, (p[CEG_COL[k]].fillna(0) <= 0) | (p['NO_CAPACITY_PCT'] >= NO_CAP_PCT),
         p[f'DAYS_ABOVE_{k}'] >= min_days, p['DAYS_WITH_REVERSE'] < MIN_REVERSE_DAYS,
         p[f'P90_PEAK_PCT_OF_CEG{SUFFIX[k]}'] >= near_pct],
        ['no reverse flow', 'no published capacity', 'above capacity', 'rare reverse flow', 'near capacity'],
        default='spare capacity'), index=p.index)


p['STATUS'] = status_for('main')
p['STATUS_DETAIL'] = np.where(p['STATUS'] != 'above capacity', p['STATUS'],
                              np.where(p['PCT_ABOVE_CEG'] >= ABOVE_SHARE_PCT, 'above capacity: energy',
                                       'above capacity: peaks only'))
p['STATUS_N1'] = status_for('n1')
p['ABOVE_N1_ONLY'] = (p['STATUS'] != 'above capacity') & (p['STATUS_N1'] == 'above capacity')
p = p.sort_values('ABOVE_kWh_PER_SOLAR_SITE', ascending=False)
p.round(4).to_csv(os.path.join(OUT, 'capacity_by_postcode.csv'))

tot = p[['ANNUAL_REVERSE_MWh', 'ANNUAL_WITHIN_MWh', 'ANNUAL_ABOVE_MWh_main', 'ANNUAL_ABOVE_MWh_n1']].sum() / 1000
print('POSTCODES by status'); print(p['STATUS_DETAIL'].value_counts().to_string())
print(f"Above capacity only under N-1: {int(p['ABOVE_N1_ONLY'].sum())}")
print('\nGWh/yr, all postcodes'); print(tot.round(2).to_string())
print(f"Share of reverse energy above CEG: {100 * tot['ANNUAL_ABOVE_MWh_main'] / tot['ANNUAL_REVERSE_MWh']:.1f}% "
      f"(N-1 {100 * tot['ANNUAL_ABOVE_MWh_n1'] / tot['ANNUAL_REVERSE_MWh']:.1f}%)")
print('\nABOVE / NEAR CAPACITY'); print(p[p['STATUS'].isin(['above capacity', 'near capacity'])][[
    'MAIN_REVERSE_SITE', 'CEG_SUMMER_MW', 'P90_PEAK_MW', 'P90_PEAK_PCT_OF_CEG', 'DAYS_ABOVE_main',
    'ANNUAL_ABOVE_MWh_main', 'PCT_ABOVE_CEG', 'STATUS_DETAIL']].round(1).to_string())
print('Saved to', OUT)

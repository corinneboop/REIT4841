### Step 6.1 - Compare installed batteries (AEMO DER register) with the reverse power flow each postcode would need
### to absorb. Run 4_reverse_flow/1_reverse_flow.py first.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

np.seterr(divide='ignore', invalid='ignore')
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
S4 = os.path.join(CODE_DIR, '2_load_data', '4_reverse_flow', 'output')
sys.path.insert(0, os.path.join(CODE_DIR, '2_load_data'))
import der_register
REVERSE_FILE = os.path.join(CODE_DIR, '2_load_data', '3_postcode_matching', '5_compare_methods', 'output', 'selected', 'reverse_MW_by_postcode.csv')
START_DATE, END_DATE = '2025-07-01', '2026-06-30'

NEED_LEVELS = {'P90': 0.90, 'P75': 0.75, 'MEDIAN': 0.50}
NEED_PERCENTILE = 90
CHARGE_EFFICIENCY = 1.0
UNDER, OVER = 0.5, 2.0
DT = 0.5

der = der_register.der_by_postcode().rename(columns={
    'battery_sites': 'BATTERY_SITES', 'battery_kW': 'BATTERY_kW', 'battery_kWh': 'BATTERY_kWh',
    'solar_sites': 'SOLAR_SITES'})[['BATTERY_SITES', 'BATTERY_kW', 'BATTERY_kWh', 'SOLAR_SITES']]
cap = pd.read_csv(os.path.join(S4, 'postcode_pv_capacity.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
postcodes = cap.index.tolist()
der = der.loc[postcodes].join(cap[['SOLAR_kVA', 'SHARE_OF_QLDSOUTH']])

period = pd.date_range(START_DATE, pd.Timestamp(END_DATE) + pd.Timedelta('23h30min'), freq='30min')
rev = pd.read_csv(REVERSE_FILE, index_col='DATETIME', parse_dates=True).reindex(period)[postcodes]

daily = pd.read_csv(os.path.join(S4, 'reverse_flow_daily_by_postcode.csv'), dtype={'POSTCODE': str}, parse_dates=['DATE'])
flag = daily.pivot(index='DATE', columns='POSTCODE', values='DAY_FLAGGED')[postcodes]
days = pd.DatetimeIndex(sorted(set(period.normalize())))
ok = ~flag.reindex(days).fillna(True).astype(bool)

X = rev.values.reshape(len(days), 48, len(postcodes)) * 1000
X = np.where(ok.values[:, None, :], X, np.nan)

daily_kWh = np.nansum(X, axis=1) * DT
daily_kWh[~ok.values] = np.nan
daily_peak_kW = np.nanmax(np.where(np.isnan(X), -np.inf, X), axis=1)
daily_peak_kW[~ok.values] = np.nan
q = NEED_PERCENTILE / 100
need = pd.DataFrame({
    'DAYS_USED': ok.sum().values,
    'DAYS_WITH_REVERSE': (daily_kWh > 0).sum(axis=0),
    'ANNUAL_REVERSE_MWh': np.nansum(daily_kWh, axis=0) * 365 / ok.sum().values / 1000,
    'MEDIAN_DAILY_REVERSE_kWh': np.nanmedian(daily_kWh, axis=0),
    'NEED_POWER_kW': np.nanquantile(daily_peak_kW, q, axis=0),
}, index=postcodes)
for lab, qq in NEED_LEVELS.items():
    need[f'NEED_{lab}_kWh'] = np.nanquantile(daily_kWh, qq, axis=0)

E = der['BATTERY_kWh'].values
P = der['BATTERY_kW'].values
soc = np.zeros((len(days), len(postcodes)))
captured = np.zeros_like(soc)
peak_after = np.zeros_like(soc)
for i in range(48):
    x = np.nan_to_num(X[:, i, :])
    room = (E - soc) / (DT * CHARGE_EFFICIENCY)
    charge = np.minimum.reduce([x, np.broadcast_to(P, x.shape), np.maximum(room, 0)])
    soc += charge * DT * CHARGE_EFFICIENCY
    captured += charge * DT
    peak_after = np.maximum(peak_after, x - charge)
captured[~ok.values] = np.nan
peak_after[~ok.values] = np.nan

res = der.join(need)
res['CAPTURE_FRACTION'] = np.nansum(captured, axis=0) / np.nansum(daily_kWh, axis=0)
res['BATTERY_UTILISATION'] = np.nansum(captured, axis=0) / (E * ok.sum().values)
res['PEAK_REDUCTION'] = 1 - np.nanquantile(peak_after, q, axis=0) / res['NEED_POWER_kW'].values

for lab in NEED_LEVELS:
    res[f'COVERAGE_{lab}'] = (res['BATTERY_kWh'] / res[f'NEED_{lab}_kWh']).where(res[f'NEED_{lab}_kWh'] > 0)
    res[f'ALIGNMENT_{lab}'] = np.select(
        [res[f'NEED_{lab}_kWh'] <= 0, res[f'COVERAGE_{lab}'] < UNDER, res[f'COVERAGE_{lab}'] > OVER],
        [f'little/no need ({lab} = 0)', 'under-aligned', 'over-aligned'], default='aligned')
res['ENERGY_COVERAGE'] = res['COVERAGE_P90']
res['POWER_COVERAGE'] = res['BATTERY_kW'] / res['NEED_POWER_kW']
res['ALIGNMENT_LOG2'] = np.log2(res['ENERGY_COVERAGE']).clip(-10, 10)
res['LOCATION_QUOTIENT'] = (res['BATTERY_kWh'] / res['BATTERY_kWh'].sum()) / (
    res['ANNUAL_REVERSE_MWh'] / res['ANNUAL_REVERSE_MWh'].sum())
per_site_batt = res['BATTERY_kWh'] / res['SOLAR_SITES']
per_site_need = res['NEED_P90_kWh'] / res['SOLAR_SITES']
res['BATTERY_kWh_PER_SOLAR_SITE'] = per_site_batt
res['NEED_kWh_PER_SOLAR_SITE'] = per_site_need
res['RANK_GAP'] = per_site_batt.rank(pct=True) - per_site_need.rank(pct=True)
res['ALIGNMENT'] = res['ALIGNMENT_P90']
res = res.replace([np.inf, -np.inf], np.nan).round(4)
res.index.name = 'POSTCODE'
res.to_csv(os.path.join(OUT, 'battery_alignment_by_postcode.csv'))

for lab in NEED_LEVELS:
    c = res[f'COVERAGE_{lab}'].dropna()
    tot = res.loc[c.index, 'BATTERY_kWh'].sum() / res.loc[c.index, f'NEED_{lab}_kWh'].sum()
    print(f"Need = {lab} daily reverse flow: {len(c)} postcodes with need, median coverage {c.median():.2f}, "
          f"covered (>= 1) {(c >= 1).mean():.0%}, total battery / total need {tot:.2f}; "
          + ', '.join(f'{k} {v}' for k, v in res[f'ALIGNMENT_{lab}'].value_counts().items()))
print(f"Batteries installed (mapped postcodes): {res['BATTERY_kWh'].sum() / 1000:,.1f} MWh, "
      f"{res['BATTERY_kW'].sum() / 1000:,.1f} MW; reverse flow {res['ANNUAL_REVERSE_MWh'].sum():,.0f} MWh/yr; "
      f"could be captured {np.nansum(captured) / 1000:,.0f} MWh/yr "
      f"({np.nansum(captured) / np.nansum(daily_kWh):.0%})")

colours = {'under-aligned': '#e34948', 'aligned': '#2a78d6', 'over-aligned': '#eda100'}
fig, axes = plt.subplots(1, len(NEED_LEVELS), figsize=(15, 5.4), sharey=True)
pos = res[[f'NEED_{l}_kWh' for l in NEED_LEVELS]].where(lambda x: x > 0)
lims = [max(1e-3, np.nanmin(pos.values) / 1000 / 2),
        max(np.nanmax(pos.values), res['BATTERY_kWh'].max()) / 1000 * 2]
for ax, lab in zip(axes, NEED_LEVELS):
    r = res[res[f'NEED_{lab}_kWh'] > 0]
    for cat, col in colours.items():
        q_ = r[r[f'ALIGNMENT_{lab}'] == cat]
        ax.scatter(q_[f'NEED_{lab}_kWh'] / 1000, q_['BATTERY_kWh'] / 1000, s=22, color=col, edgecolor='white',
                   linewidth=0.5, label=f'{cat} ({len(q_)})')
    for k, style in [(1, '-'), (UNDER, ':'), (OVER, ':')]:
        ax.plot(lims, [k * v for v in lims], color='#555555', linewidth=0.8, linestyle=style)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lims)
    ax.set_ylim(res.loc[res['BATTERY_kWh'] > 0, 'BATTERY_kWh'].min() / 1000 / 2, lims[1])
    name = 'median' if lab == 'MEDIAN' else lab
    ax.set_xlabel(f'Technical need: {name} daily reverse flow (MWh)')
    ax.set_title(f"Need = {name} day: {len(r)} postcodes, median coverage {r[f'COVERAGE_{lab}'].median():.2f}\n"
                 f"{(res[f'NEED_{lab}_kWh'] <= 0).sum()} postcodes with no {name} need not shown", fontsize=10, loc='left')
    ax.grid(color='#e0e0e0', linewidth=0.6, which='major')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper left', fontsize=8.5)
axes[0].set_ylabel('Installed battery storage (MWh, DER register, mean of 4 quarters)')
fig.suptitle(f'Battery storage vs reverse power flow by postcode at three battery sizes  '
             f'(dotted lines: coverage {UNDER} and {OVER})', x=0.01, ha='left', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_battery_alignment.png'), dpi=200)
plt.close(fig)
print('Saved to', OUT)

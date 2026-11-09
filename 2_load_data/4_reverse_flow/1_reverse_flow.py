### Step 4.1 - Reverse power flow by postcode (the energy a battery could store instead of sending it upstream).

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
LOAD_DIR = os.path.dirname(HERE)
CODE_DIR = os.path.dirname(LOAD_DIR)
sys.path.insert(0, LOAD_DIR)
import der_register

SELECTED = os.path.join(LOAD_DIR, '3_postcode_matching', '5_compare_methods', 'output', 'selected')
REVERSE_FILE = os.path.join(SELECTED, 'reverse_MW_by_postcode.csv')
NET_FILE = os.path.join(SELECTED, 'load_MW_by_postcode.csv')
MAPPING_FILE = os.path.join(SELECTED, 'site_postcode_mapping.csv')
SOLAR_FILE = os.path.join(LOAD_DIR, '3_postcode_matching', '2_network_matching', 'output', 'network', 'site_postcode_solar.csv')
SITE_LOAD_FILE = os.path.join(LOAD_DIR, '1_join_substation_data', 'output', 'load_MW_by_site.csv')
EXCLUDED_DAYS_FILE = os.path.join(LOAD_DIR, '1_join_substation_data', 'output', 'excluded_site_days.csv')
DAPR_VALIDATION_FILE = os.path.join(LOAD_DIR, '2_dapr_parameters', 'output', 'dapr_validation.csv')
PV_FILE = os.path.join(CODE_DIR, '1_pv_data', '1_unzip_and_join', 'output', 'QLDSOUTH_measured_final.csv')
POSTCODE_FILE = os.path.join(LOAD_DIR, 'data', 'raw_postcodes.csv')
OUT = os.path.join(HERE, 'output')

START_DATE, END_DATE = '2025-07-01', '2026-06-30'
QUANTILES = (10, 25, 75, 90)
LAT_RANGE, LON_RANGE = (-28.4, -25.8), (151.8, 153.6)
EXCLUDE_FLAGGED_DAYS = True
MIN_FLAG_FRACTION = 0.05
EXCLUDE_REVERSE_ABOVE_NCC = True
DT = 0.5
WHISKERS = 1.5

os.makedirs(OUT, exist_ok=True)

der = der_register.der_by_postcode()
cap = der[['solar_kVA', 'solar_sites']].rename(columns={'solar_kVA': 'SOLAR_kVA', 'solar_sites': 'SOLAR_SITES'})
cap.index.name = 'POSTCODE'
pc = pd.read_csv(POSTCODE_FILE, dtype={'postcode': str}, low_memory=False)
pc = pc[(pc['state'] == 'QLD') & (pc['type'] == 'Delivery Area')]
centre = pc.groupby('postcode')[['lat', 'long']].median()
in_region = centre['lat'].between(*LAT_RANGE) & centre['long'].between(*LON_RANGE)
region_kva = cap.loc[cap.index.isin(centre.index[in_region]), 'SOLAR_kVA'].sum()

rev = pd.read_csv(REVERSE_FILE, index_col='DATETIME', parse_dates=True)
net = pd.read_csv(NET_FILE, index_col='DATETIME', parse_dates=True)
postcodes = [p for p in rev.columns if p in cap.index]
cap = cap.loc[postcodes]
cap['SHARE_OF_QLDSOUTH'] = cap['SOLAR_kVA'] / region_kva
cap.to_csv(os.path.join(OUT, 'postcode_pv_capacity.csv'))
print(f"QLDSOUTH solar capacity (assumed region): {region_kva / 1000:,.0f} MVA; "
      f"{len(postcodes)} of {rev.shape[1]} postcodes with load and DER data")

period = pd.date_range(START_DATE, pd.Timestamp(END_DATE) + pd.Timedelta('23h30min'), freq='30min')
pv = pd.read_csv(PV_FILE, index_col='INTERVAL_START', parse_dates=True)['POWER_MW']
pv = pv.reindex(period)
rev = rev.reindex(period)[postcodes]
net = net.reindex(period)[postcodes]
pv_pc = pd.DataFrame(np.outer(pv.values, cap['SHARE_OF_QLDSOUTH'].values), index=period, columns=postcodes)
gross = net + pv_pc

site = pd.read_csv(SITE_LOAD_FILE, index_col='DATETIME', parse_dates=True).reindex(period)
site_ncc = pd.Series(dtype=float)
if EXCLUDE_REVERSE_ABOVE_NCC:
    if not os.path.exists(DAPR_VALIDATION_FILE):
        raise SystemExit('Run 2_load_data/2_dapr_parameters/dapr_validation.py first (needed for the NCC check).')
    site_ncc = pd.read_csv(DAPR_VALIDATION_FILE).groupby('SITE')['NCC_MVA_S2526'].sum(min_count=1)
step1 = pd.read_csv(EXCLUDED_DAYS_FILE, parse_dates=['DATE'])
bad_site_days = step1.groupby('SITE')['DATE'].apply(set).to_dict()
reverse_bad = []
for s in site.columns:
    ncc = site_ncc.get(s, np.nan)
    if pd.isna(ncc):
        continue
    x = site[s]
    above = -x > ncc
    for d in sorted(set(x.index[above].normalize())):
        on = x.index.normalize() == d
        reverse_bad.append({'SITE': s, 'DATE': d.date(), 'NCC_MVA': ncc, 'MIN_NET_MW': x[on].min(),
                            'INTERVALS_ABOVE_NCC': int(above[on].sum())})
        bad_site_days.setdefault(s, set()).add(d)
reverse_bad = pd.DataFrame(reverse_bad, columns=['SITE', 'DATE', 'NCC_MVA', 'MIN_NET_MW', 'INTERVALS_ABOVE_NCC'])
reverse_bad.to_csv(os.path.join(OUT, 'excluded_reverse_above_ncc_days.csv'), index=False)
print(f"Site-days with reverse flow above NCC (excluded): {len(reverse_bad)} at {reverse_bad['SITE'].nunique()} sites")

mapping = pd.read_csv(MAPPING_FILE, dtype={'POSTCODE': str})[['SITE', 'POSTCODE', 'POSTCODE_FRACTION']]
solar = pd.read_csv(SOLAR_FILE, dtype={'POSTCODE': str})[['SITE', 'POSTCODE', 'DWELLING_FRACTION']]
frac = mapping.merge(solar, on=['SITE', 'POSTCODE'], how='outer').fillna(0)
frac = frac[frac[['POSTCODE_FRACTION', 'DWELLING_FRACTION']].max(axis=1) >= MIN_FLAG_FRACTION]
days = pd.DatetimeIndex(sorted(set(period.normalize())))
flagged = pd.DataFrame(False, index=days, columns=postcodes)
for s, p in zip(frac['SITE'], frac['POSTCODE']):
    if p in flagged.columns and s in bad_site_days:
        flagged.loc[flagged.index.isin(list(bad_site_days[s])), p] = True

def positive_area(a, b, dt):
    """Area of the positive part of a straight line from a to b over dt hours (handles zero crossings)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = (a >= 0) & (b >= 0)
    cross = (a > 0) != (b > 0)
    area = np.where(both, (a + b) / 2 * dt, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        top = np.maximum(a, b)
        frac_ = top / (np.abs(a) + np.abs(b))
        area = np.where(cross & ~both, 0.5 * top * frac_ * dt, area)
    return np.nan_to_num(area)


def interval_energy(d):
    """Reverse-flow energy (MWh) in each half-hour. d = reverse flow (MW) read as the interval-midpoint value."""
    rect = d * DT
    left = (d.shift(1) + d) / 2
    right = (d + d.shift(-1)) / 2
    trap = (pd.DataFrame(positive_area(left.fillna(d).values, d.values, DT / 2), index=d.index, columns=d.columns)
            + pd.DataFrame(positive_area(d.values, right.fillna(d).values, DT / 2), index=d.index, columns=d.columns))
    return rect, trap.where(d.notna())


rect, trap = interval_energy(rev)

day = period.normalize()
def daily(df):
    return df.groupby(day).sum(min_count=1)

hour = pd.DataFrame(np.where(rev > 0, 1, np.nan), index=period, columns=postcodes).mul(
    (period.hour + period.minute / 60).values, axis=0)
d_rows = {
    'PV_MWh': daily(pv_pc * DT), 'NET_LOAD_MWh': daily(net * DT), 'GROSS_DEMAND_MWh': daily(gross * DT),
    'PEAK_GROSS_MW': gross.groupby(day).max(), 'MIN_NET_MW': net.groupby(day).min(),
    'REVERSE_RECT_MWh': daily(rect), 'REVERSE_TRAP_MWh': daily(trap),
    'REVERSE_HOURS': daily((rev > 0) * DT).where(rev.notna().groupby(day).all()),
    'REVERSE_PEAK_MW': rev.groupby(day).max(),
    'REVERSE_FIRST_START': hour.groupby(day).min(),
    'REVERSE_LAST_END': hour.groupby(day).max() + DT,
}
dd = pd.concat({k: v.stack(future_stack=True) for k, v in d_rows.items()}, axis=1)
dd.index.names = ['DATE', 'POSTCODE']
dd = dd.reset_index()
missing = rev.isna().groupby(day).any().stack(future_stack=True)
dd['DAY_MISSING'] = missing.values
dd['DAY_FLAGGED'] = flagged.stack(future_stack=True).values | dd['DAY_MISSING']
dd = dd.merge(cap[['SOLAR_kVA', 'SOLAR_SITES']], left_on='POSTCODE', right_index=True)
e = dd['REVERSE_TRAP_MWh']
dd['REVERSE_kWh_per_kVA'] = e * 1000 / dd['SOLAR_kVA']
dd['REVERSE_PCT_OF_PV'] = 100 * e / dd['PV_MWh']
dd['REVERSE_kWh_per_SOLAR_SITE'] = e * 1000 / dd['SOLAR_SITES']
dd['PV_PENETRATION'] = dd['SOLAR_kVA'] / 1000 / dd['PEAK_GROSS_MW']
dd.round(4).to_csv(os.path.join(OUT, 'reverse_flow_daily_by_postcode.csv'), index=False)

use = dd[~dd['DAY_FLAGGED']] if EXCLUDE_FLAGGED_DAYS else dd
measures = ['PV_MWh', 'GROSS_DEMAND_MWh', 'PV_PENETRATION', 'REVERSE_RECT_MWh', 'REVERSE_TRAP_MWh',
            'REVERSE_HOURS', 'REVERSE_PEAK_MW', 'REVERSE_kWh_per_kVA', 'REVERSE_PCT_OF_PV', 'REVERSE_kWh_per_SOLAR_SITE']
g = use.groupby('POSTCODE')
summary = pd.concat({
    'DAYS': g.size(),
    'DAYS_WITH_REVERSE': g['REVERSE_TRAP_MWh'].apply(lambda s: (s > 0).sum()),
    **{f'{c}_MEDIAN': g[c].median() for c in measures},
    'REVERSE_TRAP_MWh_MEAN': g['REVERSE_TRAP_MWh'].mean(),
    **{f'REVERSE_TRAP_MWh_P{q}': g['REVERSE_TRAP_MWh'].quantile(q / 100) for q in QUANTILES},
}, axis=1).round(3)
summary = summary.join(cap[['SOLAR_kVA', 'SOLAR_SITES']])
summary.to_csv(os.path.join(OUT, 'reverse_flow_summary_by_postcode.csv'))

n_days = use.groupby('POSTCODE').size()
annual = use.groupby('POSTCODE')['REVERSE_TRAP_MWh'].sum() * 365 / n_days / 1000
print(f"Postcode-days flagged: {dd['DAY_FLAGGED'].sum():,} of {len(dd):,} "
      f"({dd['DAY_MISSING'].sum():,} with missing half-hours)")
print(f"Reverse flow, usable days scaled to a year: {annual.sum():.1f} GWh/yr "
      f"(trapezoid; rectangle {use['REVERSE_RECT_MWh'].sum() / use['REVERSE_TRAP_MWh'].sum():.3f} x trapezoid)")
print(f"Postcodes with any reverse flow: {(annual > 0).sum()} of {len(annual)}; top 5 (GWh/yr):")
print(annual.sort_values(ascending=False).head().round(1).to_string())

ok_pd = ~flagged.reindex(columns=postcodes).values & ~missing.unstack().reindex(index=days, columns=postcodes).values
e_day = daily(trap).reindex(days)[postcodes].where(ok_pd)
n_ok = e_day.notna().sum(axis=1)
net_e = (e_day.sum(axis=1) * len(postcodes) / n_ok.replace(0, np.nan)) / 1000
ok_hh = pd.DataFrame(np.repeat(ok_pd, 48, axis=0), index=period, columns=postcodes)
r_ok = rev.where(ok_hh)
net_p = (r_ok.sum(axis=1) * len(postcodes) / r_ok.notna().sum(axis=1).replace(0, np.nan)).groupby(day).max()
mon = pd.DataFrame({'E': net_e, 'P': net_p.reindex(days)}, index=days).dropna()
mon['MONTH'] = mon.index.to_period('M')
order = sorted(mon['MONTH'].unique())
lab = [m.strftime('%b\n%Y') if m.month in (7, 1) else m.strftime('%b') for m in order]
q = lambda x, v: x.quantile(v)
tab = mon.groupby('MONTH').agg(DAYS=('E', 'size'), TOTAL_GWh=('E', 'sum'),
                               DAILY_MIN_GWh=('E', 'min'), DAILY_P25_GWh=('E', lambda x: q(x, .25)),
                               DAILY_MEDIAN_GWh=('E', 'median'), DAILY_MEAN_GWh=('E', 'mean'),
                               DAILY_P75_GWh=('E', lambda x: q(x, .75)), DAILY_P90_GWh=('E', lambda x: q(x, .9)),
                               DAILY_MAX_GWh=('E', 'max'), PEAK_MIN_MW=('P', 'min'), PEAK_MEDIAN_MW=('P', 'median'),
                               PEAK_MAX_MW=('P', 'max'))
tab['TOTAL_GWh'] = tab['TOTAL_GWh'] * tab.index.days_in_month / tab['DAYS']
tab['SHARE_OF_YEAR_PCT'] = 100 * tab['TOTAL_GWh'] / tab['TOTAL_GWh'].sum()
tab['MAX_DAY'] = mon.groupby('MONTH')['E'].idxmax().dt.strftime('%Y-%m-%d')
tab['PEAK_MAX_DAY'] = mon.groupby('MONTH')['P'].idxmax().dt.strftime('%Y-%m-%d')
for col, pre, unit in [('E', 'DAILY', 'GWh'), ('P', 'PEAK', 'MW')]:
    st = {m: boxplot_stats(mon.loc[mon['MONTH'] == m, col].values, whis=WHISKERS)[0] for m in tab.index}
    tab[f'{pre}_LOWER_WHISKER_{unit}'] = [st[m]['whislo'] for m in tab.index]
    tab[f'{pre}_UPPER_WHISKER_{unit}'] = [st[m]['whishi'] for m in tab.index]
    tab[f'{pre}_OUTLIER_DAYS'] = [len(st[m]['fliers']) for m in tab.index]
tab.round(3).to_csv(os.path.join(OUT, 'monthly_reverse_energy.csv'))
print('\nMonthly network reverse flow:'); print(tab.round(2).to_string())

BLUE, INK, MUTED, GRID = '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df'
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
x = np.arange(len(order))
ax = axes[0]
ax.bar(x, tab['TOTAL_GWh'].values, width=0.7, color=BLUE)
for xi, v, sh in zip(x, tab['TOTAL_GWh'].values, tab['SHARE_OF_YEAR_PCT'].values):
    ax.text(xi, v + 0.6, f'{sh:.0f}%', ha='center', va='bottom', fontsize=8, color=MUTED)
ax.set_ylabel('Reverse energy (GWh per month)')
ax.set_title(f"(a) Monthly total ({tab['TOTAL_GWh'].sum():.0f} GWh/yr; label = % of year)", loc='left', fontsize=10)


def boxes(ax, col, unit, title):
    data = [mon.loc[mon['MONTH'] == m, col].values for m in order]
    ax.boxplot(data, positions=x, widths=0.55, whis=WHISKERS, patch_artist=True, showmeans=True,
               flierprops=dict(marker='o', markersize=3.5, markerfacecolor='none', markeredgecolor='#777777'),
               boxprops=dict(facecolor='#cde2fb', edgecolor=BLUE, linewidth=1.2),
               medianprops=dict(color=BLUE, linewidth=2), whiskerprops=dict(color=MUTED, linewidth=1),
               capprops=dict(color=MUTED, linewidth=1),
               meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor=INK, markersize=5))
    ax.set_ylabel(unit); ax.set_title(title, loc='left', fontsize=10)


boxes(axes[1], 'E', 'Daily reverse energy (GWh/day)', '(b) Daily reverse energy: range within each month')
boxes(axes[2], 'P', 'Daily peak reverse flow (MW)', '(c) Daily peak reverse flow: range within each month')
for ax in axes:
    ax.set_xticks(x, lab, fontsize=8.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
fig.text(0.01, 0.01, 'Network = all 199 postcodes (usable postcode-days, scaled to the full network). '
         'Box = P25-P75, line = median, diamond = mean, whiskers = 1.5 x IQR, circles = days beyond the whiskers.', fontsize=8.5, color=MUTED)
fig.suptitle('Reverse power flow across the Energex network by month, Jul 2025 - Jun 2026', x=0.01, ha='left', fontsize=12)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(os.path.join(OUT, 'fig_3_monthly_reverse_energy.png'), dpi=200)
plt.close(fig)
print('Saved to', OUT)

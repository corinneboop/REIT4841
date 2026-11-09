### Step 4.2 - Average postcode day: net load, reverse power flow and rooftop solar (Jul 2025 - Jun 2026, no seasonal
### split). Run 1_reverse_flow.py first.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD_DIR = os.path.dirname(HERE)
CODE = os.path.dirname(LOAD_DIR)
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
SELECTED = os.path.join(LOAD_DIR, '3_postcode_matching', '5_compare_methods', 'output', 'selected')
S4 = os.path.join(LOAD_DIR, '4_reverse_flow', 'output')
PV = os.path.join(CODE, '1_pv_data', '1_unzip_and_join', 'output', 'QLDSOUTH_measured_final.csv')

START, END = '2025-07-01', '2026-06-30'
MIN_DAYS = 10
C_SOLAR, C_NET, C_GROSS, C_REV = '#eda100', '#2a78d6', '#7a7a7a', '#c0392b'

period = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta('23h30min'), freq='30min')
days = pd.DatetimeIndex(sorted(set(period.normalize())))
D = len(days)
d4 = pd.read_csv(os.path.join(S4, 'reverse_flow_daily_by_postcode.csv'), dtype={'POSTCODE': str}, parse_dates=['DATE'])
use = d4[~d4['DAY_FLAGGED']]
n_rev = (use['REVERSE_TRAP_MWh'] > 0).groupby(use['POSTCODE']).sum()
cap = pd.read_csv(os.path.join(S4, 'postcode_pv_capacity.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
postcodes = n_rev.index[n_rev >= MIN_DAYS].tolist()
P = len(postcodes)
kva = cap.loc[postcodes, 'SOLAR_kVA'].values

net = pd.read_csv(os.path.join(SELECTED, 'load_MW_by_postcode.csv'), index_col='DATETIME', parse_dates=True)
rev = pd.read_csv(os.path.join(SELECTED, 'reverse_MW_by_postcode.csv'), index_col='DATETIME', parse_dates=True)
pv = pd.read_csv(PV, index_col='INTERVAL_START', parse_dates=True)['POWER_MW']
pv = pv.reindex(period).values.reshape(D, 48)
N = net.reindex(period)[postcodes].values.reshape(D, 48, P)
R = rev.reindex(period)[postcodes].values.reshape(D, 48, P)
S = pv[:, :, None] * cap.loc[postcodes, 'SHARE_OF_QLDSOUTH'].values[None, None, :]
ok = ~d4.pivot(index='DATE', columns='POSTCODE', values='DAY_FLAGGED').reindex(index=days, columns=postcodes) \
    .fillna(True).astype(bool).values
ok &= ~np.isnan(N).any(axis=1) & ~np.isnan(R).any(axis=1) & ~np.isnan(pv).any(axis=1)[:, None]
hh = np.arange(48) / 2 + 0.25
k = 1000 / kva[None, None, :]
Nn, Rn, Sn = N * k, R * k, S * k
print(f"{P} postcodes with >= 10 reverse-flow days; {ok.sum():,} usable postcode-days")


def profile(mask, n, r, s):
    """Mean profiles over the postcode-days in mask (D, P)."""
    sel = lambda x: np.transpose(x, (0, 2, 1))[mask]
    n_, r_, s_ = sel(n), sel(r), sel(s)
    return pd.DataFrame({'hour': hh, 'solar': s_.mean(0), 'net': n_.mean(0), 'net_p25': np.percentile(n_, 25, 0),
                         'net_p75': np.percentile(n_, 75, 0), 'net_median': np.median(n_, 0), 'gross': (n_ + s_).mean(0), 'reverse': r_.mean(0),
                         'share_reverse': (r_ > 0).mean(0)}), len(n_)


def hm(x):
    return f'{int(x):02d}:{int(round((x % 1) * 60)):02d}'


def window(p):
    on = p['reverse'].values > 0.02 * p['reverse'].max()
    return (hm(p['hour'][np.argmax(on)] - 0.25), hm(p['hour'][len(on) - 1 - np.argmax(on[::-1])] + 0.25)) if on.any() else None


def style(ax, unit):
    ax.axhline(0, color='#333333', lw=0.9)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3), [f'{h:02d}:00' for h in range(0, 25, 3)], fontsize=8)
    ax.set_ylabel(unit, fontsize=9)
    ax.grid(color='#e6e6e6', lw=0.6); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def panel(ax, p, unit, band=True):
    if band:
        ax.fill_between(p['hour'], p['net_p25'], p['net_p75'], color=C_NET, alpha=0.12, lw=0, label='Net load P25-P75')
    ax.plot(p['hour'], p['gross'], color=C_GROSS, ls='--', lw=1.5, label='Gross demand (net + solar)')
    ax.plot(p['hour'], p['solar'], color=C_SOLAR, lw=2.2, label='Rooftop solar')
    ax.plot(p['hour'], p['net'], color=C_NET, lw=2.4, label='Net load (mean)')
    ax.plot(p['hour'], p['net_median'], color=C_NET, lw=1.4, ls='--', label='Net load (median)')
    ax.fill_between(p['hour'], -p['reverse'], 0, color=C_REV, alpha=0.4, lw=0, label='Reverse power flow (R_p, below 0)')
    style(ax, unit)


rows = []
p_all, n = profile(ok, Nn, Rn, Sn)
rows.append(p_all.assign(group='all postcodes', postcode_days=n))
fig, ax = plt.subplots(figsize=(9, 5.8))
panel(ax, p_all, 'kW per kVA of rooftop solar')
w = window(p_all)
ax.set_title(f"Mean of {n:,} postcode-days\nmean reverse flow {w[0]}-{w[1]}, peak "
             f"{p_all['reverse'].max():.3f} kW/kVA at {hm(p_all.loc[p_all['reverse'].idxmax(), 'hour'])}\n"
             f"{100 * p_all['share_reverse'].max():.0f}% of postcode-days have reverse flow at that time", loc='left', fontsize=9.5)
ax.set_xlabel('Time of day (AEST)', fontsize=9)
print(f"All days: mean reverse flow {w[0]}-{w[1]}, peak {p_all['reverse'].max():.3f} kW/kVA, solar peak "
      f"{p_all['solar'].max():.3f} kW/kVA, min mean net {p_all['net'].min():.3f} kW/kVA")
ax.legend(loc='upper left', frameon=False, fontsize=8.5)
fig.suptitle(f'Average postcode day: net load, reverse power flow and rooftop solar\n'
             f'{P} postcodes, Jul 2025 - Jun 2026, each scaled by its rooftop solar.\nMean above median: a few '
             f'business-heavy postcodes have high load per kVA.', x=0.01, ha='left', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(os.path.join(OUT, 'fig_average_day.png'), dpi=200)
plt.close(fig)
pd.concat(rows).round(4).to_csv(os.path.join(OUT, 'average_profiles.csv'), index=False)
print('Saved to', OUT)

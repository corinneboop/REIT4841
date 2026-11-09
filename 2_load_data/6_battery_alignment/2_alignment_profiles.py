### Step 6.2 - Net load, reverse power flow and rooftop solar through the day, by battery alignment.
### Run 1_battery_alignment.py (same folder) and 5_frequency_magnitude first.

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
S5 = os.path.join(LOAD_DIR, '5_frequency_magnitude', 'output')
PV = os.path.join(CODE, '1_pv_data', '1_unzip_and_join', 'output', 'QLDSOUTH_measured_final.csv')

START, END = '2025-07-01', '2026-06-30'
C_SOLAR, C_NET, C_GROSS, C_REV = '#eda100', '#2a78d6', '#7a7a7a', '#c0392b'
ALIGN_COLOURS = {'under-aligned': '#e34948', 'aligned': '#2a78d6', 'over-aligned': '#eda100'}

period = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta('23h30min'), freq='30min')
days = pd.DatetimeIndex(sorted(set(period.normalize())))
D = len(days)
fm = pd.read_csv(os.path.join(S5, 'frequency_magnitude_by_postcode.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
al = pd.read_csv(os.path.join(OUT, 'battery_alignment_by_postcode.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
cap = pd.read_csv(os.path.join(S4, 'postcode_pv_capacity.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
postcodes = fm.index[fm['ANALYSED']].tolist()
P = len(postcodes)
kva = cap.loc[postcodes, 'SOLAR_kVA'].values

net = pd.read_csv(os.path.join(SELECTED, 'load_MW_by_postcode.csv'), index_col='DATETIME', parse_dates=True)
rev = pd.read_csv(os.path.join(SELECTED, 'reverse_MW_by_postcode.csv'), index_col='DATETIME', parse_dates=True)
pv = pd.read_csv(PV, index_col='INTERVAL_START', parse_dates=True)['POWER_MW']
pv = pv.reindex(period).values.reshape(D, 48)
N = net.reindex(period)[postcodes].values.reshape(D, 48, P)
R = rev.reindex(period)[postcodes].values.reshape(D, 48, P)
S = pv[:, :, None] * cap.loc[postcodes, 'SHARE_OF_QLDSOUTH'].values[None, None, :]
d4 = pd.read_csv(os.path.join(S4, 'reverse_flow_daily_by_postcode.csv'), dtype={'POSTCODE': str}, parse_dates=['DATE'])
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

cls = al['ALIGNMENT_P90'].reindex(postcodes).values
fig, ax = plt.subplots(figsize=(9, 5.4))
for c, col in ALIGN_COLOURS.items():
    m = ok & (cls == c)[None, :]
    if not m.any():
        continue
    p, n = profile(m, Nn, Rn, Sn)
    rows.append(p.assign(group=c, postcode_days=n))
    npc = int((cls == c).sum())
    ax.plot(p['hour'], p['net'], color=col, lw=2.2, label=f'{c}: net load ({npc} postcodes)')
    ax.plot(p['hour'], -p['reverse'], color=col, lw=1.6, ls='--', label=f'{c}: reverse flow (below 0)')
ax.plot(p_all['hour'], p_all['solar'], color=C_SOLAR, lw=1.4, alpha=0.8, label='Rooftop solar (all)')
style(ax, 'kW per kVA of rooftop solar')
ax.set_xlabel('Time of day (AEST)', fontsize=9)
ax.legend(loc='upper left', frameon=False, fontsize=7.5)
fig.suptitle('Mean postcode net load and reverse flow by battery alignment (P90 coverage: under < 0.5, over > 2)',
             x=0.01, ha='left', fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(OUT, 'fig_alignment_profiles.png'), dpi=200)
plt.close(fig)
pd.concat(rows).round(4).to_csv(os.path.join(OUT, 'alignment_profiles.csv'), index=False)

cp5 = pd.read_csv(os.path.join(S5, 'capacity_by_postcode.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
c5 = pd.read_csv(os.path.join(S5, 'capacity_daily_by_postcode.csv'), dtype={'POSTCODE': str}, parse_dates=['DATE'])
ceg_day = c5.pivot(index='DATE', columns='POSTCODE', values='CEG_MW_main').reindex(days)
site_load = pd.read_csv(os.path.join(LOAD_DIR, '1_join_substation_data', 'output', 'load_MW_by_site.csv'),
                        index_col='DATETIME', parse_dates=True).reindex(period)
mp7 = pd.read_csv(os.path.join(SELECTED, 'site_postcode_mapping.csv'), dtype={'POSTCODE': str})
SS7 = mp7.pivot_table(index='SITE', columns='POSTCODE', values='SOLAR_SHARE', aggfunc='sum').fillna(0)
SS7 = SS7[SS7.index.isin(site_load.columns)]
dapr7 = pd.read_csv(os.path.join(LOAD_DIR, '2_dapr_parameters', 'output', 'dapr_parameters_by_bus.csv')).merge(
    pd.read_csv(os.path.join(LOAD_DIR, '2_dapr_parameters', 'output', 'dapr_validation.csv'))[['CODE', 'SITE']], on='CODE')
CEGC = 'CAPACITY_FOR_EMBEDDED_GENERATION_MVA__'
ceg7 = {c: dapr7.groupby('SITE')[CEGC + c].apply(lambda x: pd.to_numeric(x, errors='coerce').sum(min_count=1))
        for c in ('S2526', 'W2025', 'W2026')}


def above_capacity(pcode, d):
    """Half-hourly reverse flow of postcode pcode above its substations' CEG on day index d (MW)."""
    w = SS7[pcode][SS7[pcode] > 0] if pcode in SS7.columns else pd.Series(dtype=float)
    day = days[d]
    col = 'S2526' if day.month in (9, 10, 11, 12, 1, 2) else ('W2025' if day.year == 2025 else 'W2026')
    out = np.zeros(48)
    for s_, ws in w.items():
        c = ceg7[col].get(s_, np.nan)
        if pd.isna(c) or c <= 0:
            continue
        rs = (-site_load[s_].clip(upper=0)).fillna(0).values.reshape(D, 48)[d]
        out += ws * np.clip(rs - c, 0, None)
    return out
sel = al.join(cp5[['STATUS_DETAIL', 'MAIN_REVERSE_SITE', 'MAIN_SITE_SHARE_PCT', 'PCT_ABOVE_CEG', 'DAYS_ABOVE_main']])
sel = sel[sel.index.isin(postcodes) & (sel['ALIGNMENT_P90'] == 'under-aligned')
          & (sel['STATUS_DETAIL'] == 'above capacity: energy')].sort_values('PCT_ABOVE_CEG', ascending=False)
print(f"\nUnder-aligned and above capacity (energy): {', '.join(sel.index)}")
E = np.nansum(R, axis=1) * 0.5
crow, drow = [], []
n_p = len(sel)
nc = 3
nr = int(np.ceil((n_p + 1) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(15, 5.2 * nr))
for ax, (pcode, r) in zip(axes.flat, sel.iterrows()):
    j = postcodes.index(pcode)
    m = np.zeros((D, P), bool); m[:, j] = ok[:, j]
    p, n = profile(m, N, R, S)
    e = np.where(ok[:, j], E[:, j], np.nan)
    p90 = np.nanquantile(e, 0.9)
    i90 = int(np.nanargmin(np.abs(e - p90)))
    cap_d = ceg_day[pcode].iloc[i90]
    ab = above_capacity(pcode, i90)
    main_ceg = ceg7['S2526' if days[i90].month in (9, 10, 11, 12, 1, 2) else ('W2025' if days[i90].year == 2025 else 'W2026')].get(r['MAIN_REVERSE_SITE'], np.nan)
    top = np.where(e >= p90)[0]
    drow += [{'POSTCODE': pcode, 'DATE': days[t].date(), 'REVERSE_MWh': round(e[t], 2)} for t in top]
    crow.append(p.assign(POSTCODE=pcode, days=n, p90_day=str(days[i90].date()), p90_solar=S[i90, :, j],
                         p90_net=N[i90, :, j], p90_reverse=R[i90, :, j], p90_above_capacity=ab, capacity_MW=cap_d))
    ax.plot(p['hour'], p['solar'], color=C_SOLAR, lw=2.2, label='Rooftop solar (mean day)')
    ax.plot(p['hour'], p['net'], color=C_NET, lw=2.2, label='Net load (mean day)')
    ax.fill_between(p['hour'], -p['reverse'], 0, color=C_REV, alpha=0.35, lw=0, label='Reverse flow (mean day)')
    ax.plot(p['hour'], S[i90, :, j], color=C_SOLAR, lw=1.2, ls=':', label='Rooftop solar (P90 day)')
    ax.plot(p['hour'], N[i90, :, j], color=C_NET, lw=1.2, ls=':', label='Net load (P90 day)')
    ax.plot(p['hour'], -R[i90, :, j], color=C_REV, lw=1.6, ls='--', label='Reverse flow (P90 day)')
    ax.fill_between(p['hour'], -ab, 0, color='#7b1010', alpha=0.85, lw=0, label='P90 day: reverse flow above hosting capacity')
    style(ax, 'MW')
    ax.set_title(f"{pcode} ({r['MAIN_REVERSE_SITE']}, {r['MAIN_SITE_SHARE_PCT']:.0f}% of its reverse flow)\n"
                 f"P90 coverage {r['COVERAGE_P90']:.2f}; {r['PCT_ABOVE_CEG']:.0f}% of reverse energy above capacity, "
                 f"{int(r['DAYS_ABOVE_main'])} days\nP90 day {days[i90]:%d %b %Y}: {e[i90]:.1f} MWh reverse, "
                 f"{ab.sum() * 0.5:.1f} MWh above capacity\n{r['MAIN_REVERSE_SITE']} hosting capacity (CEG) "
                 f"{main_ceg:.1f} MW",
                 loc='left', fontsize=8.5)
mon = pd.DataFrame(drow)
ax = axes.flat[n_p]
if len(mon):
    mon['MONTH'] = pd.to_datetime(mon['DATE']).dt.month
    order = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    cnt = mon.groupby(['MONTH', 'POSTCODE']).size().unstack(fill_value=0).reindex(order, fill_value=0)
    bottom = np.zeros(12)
    for c_, pc_ in zip(['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'], cnt.columns):
        ax.bar(range(12), cnt[pc_].values, bottom=bottom, color=c_, label=pc_); bottom += cnt[pc_].values
    ax.set_xticks(range(12), ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], fontsize=8)
    ax.set_ylabel('Days at or above the postcode P90', fontsize=9)
    ax.set_title('When the P90 days occur (Jul 2025 - Jun 2026)', loc='left', fontsize=9)
    ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e6e6', lw=0.6); ax.set_axisbelow(True)
for ax in axes.flat[n_p + 1:]:
    ax.set_visible(False)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=8.5)
fig.suptitle('Under-aligned postcodes above hosting capacity: net load, reverse power flow and rooftop solar '
             '(mean day and representative P90 day)', x=0.01, ha='left', fontsize=11.5)
fig.tight_layout(rect=(0, 0.06, 1, 0.96)); fig.savefig(os.path.join(OUT, 'fig_over_capacity_under_aligned.png'), dpi=200)
plt.close(fig)
if crow:
    pd.concat(crow).round(4).to_csv(os.path.join(OUT, 'over_capacity_profiles.csv'), index=False)
    mon.to_csv(os.path.join(OUT, 'over_capacity_p90_days.csv'), index=False)
    print(mon.groupby(['POSTCODE', pd.to_datetime(mon['DATE']).dt.month]).size().unstack(fill_value=0).to_string())
print('Saved to', OUT)

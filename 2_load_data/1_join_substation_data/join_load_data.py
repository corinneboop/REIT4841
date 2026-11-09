### STEP 1 - Join the Energex zone substation load files and find the periods to leave out.

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, 'raw_load_data')
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'output')
VALUE_COLUMN = 'MW'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
ZERO_RUN_HOURS = 2
FLAT_RUN_HOURS = 4
DAYLIGHT_HOURS = range(6, 18)
NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4]
NIGHT_REVERSE_HOURS = 1
EXCLUDE_SITES = {'Nth Pine Dam': 'Dam pumping station: 0 MW for 57% of the year, median 0.00 MW, max 1.3 MW; '
                                 'no residential load to analyse',
                 'Amberley': 'Small flows recorded as zero: 37% of 09:00-14:30 readings are exactly 0.00 MW vs ~1.4% '
                             'per adjacent 0.01 MW step, so reverse flow cannot be measured'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def site_name(name):
    """Remove the bus number: 'Bald Hills Bus 2' -> 'Bald Hills' (same rule as 2_dapr_parameters)."""
    return re.sub(r'\s+Bus\s*\d+$', '', str(name)).strip()


series, summary = {}, []
for filename in sorted(os.listdir(INPUT_FOLDER)):
    if not filename.endswith('.csv'):
        continue
    bus = filename.split('_EGX_')[0]
    df = pd.read_csv(os.path.join(INPUT_FOLDER, filename), encoding='utf-8-sig', dtype=str)
    if df.empty:
        summary.append({'SUBSTATION': bus, 'ROWS': 0, 'NOTE': 'file has no data'})
        continue
    time = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
    values = pd.to_numeric(df[VALUE_COLUMN], errors='coerce')
    s = pd.Series(values.values, index=time).sort_index()
    s = s[~s.index.duplicated()]
    series[bus] = s
    summary.append({'SUBSTATION': bus, 'SITE': site_name(bus), 'ROWS': len(df),
                    'FIRST': s.index.min(), 'LAST': s.index.max(),
                    'BLANK': int(s.isna().sum()), 'ZERO': int((s == 0).sum()), 'NEGATIVE': int((s < 0).sum())})

start = min(s.index.min() for s in series.values())
end = max(s.index.max() for s in series.values())
full_index = pd.date_range(start, end, freq='30min', name='DATETIME')
print(f"Timeline: {start} to {end} ({len(full_index)} half-hours)")
wide = pd.DataFrame({name: s.reindex(full_index) for name, s in series.items()})
summary = pd.DataFrame(summary)
summary['ABSENT'] = summary['SUBSTATION'].map({b: int((~full_index.isin(s.index)).sum()) for b, s in series.items()})
gaps = []
for bus in wide.columns:
    miss = wide[bus].isna()
    for _, g in miss[miss].groupby((~miss).cumsum()):
        absent = ~g.index.isin(series[bus].index)
        gaps.append({'SUBSTATION': bus, 'GAP_START': g.index[0], 'GAP_END': g.index[-1], 'INTERVALS': len(g),
                     'HOURS': len(g) / 2, 'TYPE': 'absent' if absent.all() else 'blank' if not absent.any() else 'both'})
gaps = pd.DataFrame(gaps, columns=['SUBSTATION', 'GAP_START', 'GAP_END', 'INTERVALS', 'HOURS', 'TYPE'])

site_of = pd.Series({c: site_name(c) for c in wide.columns})
n_bus = site_of.value_counts()
site = wide.T.groupby(site_of).sum(min_count=1).T
site = site.where(wide.notna().T.groupby(site_of).sum().T.eq(n_bus[site.columns], axis=1))
site_buses = site_of.groupby(site_of).apply(lambda s: ' + '.join(s.index))
print(f"{wide.shape[1]} buses -> {site.shape[1]} zone substations "
      f"({(site_of.value_counts() > 1).sum()} with more than one bus)")

runs = []
for s_name in site.columns:
    x = site[s_name]
    rid = (x != x.shift()).cumsum()
    size = rid.map(rid.value_counts())
    for _, run in x[size / 2 >= min(ZERO_RUN_HOURS, FLAT_RUN_HOURS)].groupby(rid):
        value, n = run.iloc[0], len(run)
        if pd.isna(value):
            continue
        kind = 'zero_run' if value == 0 else 'flat_run'
        if n / 2 < (ZERO_RUN_HOURS if kind == 'zero_run' else FLAT_RUN_HOURS):
            continue
        i0, i1 = x.index.get_loc(run.index[0]), x.index.get_loc(run.index[-1])
        runs.append({'SITE': s_name, 'BUSES': site_buses[s_name], 'TYPE': kind,
                     'RUN_START': run.index[0], 'RUN_END': run.index[-1],
                     'INTERVALS': n, 'HOURS': n / 2, 'VALUE_MW': value,
                     'MW_BEFORE': x.iloc[i0 - 1] if i0 > 0 else np.nan,
                     'MW_AFTER': x.iloc[i1 + 1] if i1 + 1 < len(x) else np.nan,
                     'DAYLIGHT_INTERVALS': int(np.isin(run.index.hour, list(DAYLIGHT_HOURS)).sum()),
                     'REASON': 'site left out: ' + EXCLUDE_SITES[s_name] if s_name in EXCLUDE_SITES else
                               (f'zero for {n / 2:g} h: switched out or metering lost' if kind == 'zero_run'
                                else f'{value:.2f} MW unchanged for {n / 2:g} h: meter frozen')})
for s_name in site.columns:
    if s_name in EXCLUDE_SITES:
        continue
    x = site[s_name]
    night = x[np.isin(x.index.hour, NIGHT_HOURS)]
    n_rev = (night < 0).groupby(night.index.normalize()).sum()
    for day, n in n_rev[n_rev / 2 >= NIGHT_REVERSE_HOURS].items():
        d = x.loc[day:day + pd.Timedelta('23h30min')]
        runs.append({'SITE': s_name, 'BUSES': site_buses[s_name], 'TYPE': 'night_reverse',
                     'RUN_START': d.index[0], 'RUN_END': d.index[-1], 'INTERVALS': len(d), 'HOURS': len(d) / 2,
                     'VALUE_MW': d.min(), 'MW_BEFORE': np.nan, 'MW_AFTER': np.nan,
                     'DAYLIGHT_INTERVALS': int(np.isin(d.index.hour, list(DAYLIGHT_HOURS)).sum()),
                     'REASON': f'reverse flow at night for {n / 2:g} h (min {d.min():.1f} MW): not attributable to rooftop '
                               f'solar (metering/switching error, load transfer or battery/generator export)'})
flags = pd.DataFrame(runs, columns=['SITE', 'BUSES', 'TYPE', 'RUN_START', 'RUN_END', 'INTERVALS', 'HOURS',
                                    'VALUE_MW', 'MW_BEFORE', 'MW_AFTER', 'DAYLIGHT_INTERVALS', 'REASON'])

day_rows = []
for r in flags[~flags['SITE'].isin(list(EXCLUDE_SITES))].itertuples():
    idx = pd.date_range(r.RUN_START, r.RUN_END, freq='30min')
    for d in sorted(set(idx[np.isin(idx.hour, list(DAYLIGHT_HOURS))].normalize())):
        day_rows.append({'SITE': r.SITE, 'DATE': d.strftime('%Y-%m-%d'), 'REASON': r.REASON})
excluded_days = (pd.DataFrame(day_rows, columns=['SITE', 'DATE', 'REASON'])
                 .groupby(['SITE', 'DATE'])['REASON'].agg('; '.join).reset_index())
excluded_sites = pd.DataFrame([{'SITE': k, 'BUSES': site_buses.get(k, k), 'REASON': v}
                               for k, v in EXCLUDE_SITES.items()], columns=['SITE', 'BUSES', 'REASON'])

clean = site.drop(columns=[s for s in EXCLUDE_SITES if s in site.columns])
for r in flags[flags['SITE'].isin(clean.columns)].itertuples():
    clean.loc[r.RUN_START:r.RUN_END, r.SITE] = np.nan
print(f"Cleaned site load: {clean.shape[1]} sites; {int(clean.isna().sum().sum()):,} half-hours blanked at "
      f"{int(clean.isna().any().sum())} sites")

for col in ['FIRST', 'LAST']:
    summary[col] = pd.to_datetime(summary[col]).dt.strftime(DATE_FORMAT)
out = wide.copy()
out.index = out.index.strftime(DATE_FORMAT)
out.to_csv(os.path.join(OUTPUT_FOLDER, f'load_{VALUE_COLUMN}_all_substations.csv'))
clean_out = clean.copy()
clean_out.index = clean_out.index.strftime(DATE_FORMAT)
clean_out.to_csv(os.path.join(OUTPUT_FOLDER, f'load_{VALUE_COLUMN}_by_site.csv'))
site_buses.drop([s for s in EXCLUDE_SITES if s in site_buses.index]).rename('BUSES').to_csv(
    os.path.join(OUTPUT_FOLDER, 'site_buses.csv'), index_label='SITE')
summary.to_csv(os.path.join(OUTPUT_FOLDER, 'load_substation_summary.csv'), index=False)
for col in ['GAP_START', 'GAP_END']:
    gaps[col] = pd.to_datetime(gaps[col]).dt.strftime(DATE_FORMAT)
gaps.to_csv(os.path.join(OUTPUT_FOLDER, 'load_missing_intervals.csv'), index=False)
plot_flags = flags.copy()
for col in ['RUN_START', 'RUN_END']:
    flags[col] = flags[col].dt.strftime(DATE_FORMAT)
flags.round(3).to_csv(os.path.join(OUTPUT_FOLDER, 'load_quality_flags.csv'), index=False)
excluded_days.to_csv(os.path.join(OUTPUT_FOLDER, 'excluded_site_days.csv'), index=False)
excluded_sites.to_csv(os.path.join(OUTPUT_FOLDER, 'excluded_sites.csv'), index=False)

ZERO, FLAT, NIGHT, INK, MUTED, GRID = '#eb6834', '#6250d6', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df'
COLOUR = {'zero_run': ZERO, 'flat_run': FLAT, 'night_reverse': NIGHT}
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
order = plot_flags.groupby('SITE')['HOURS'].sum().sort_values(ascending=False).index
with PdfPages(os.path.join(OUTPUT_FOLDER, 'excluded_runs_plots.pdf')) as pdf:
    for s_name in order:
        rr = plot_flags[plot_flags['SITE'] == s_name].sort_values('HOURS', ascending=False)
        x = site[s_name]
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1], hspace=0.6, wspace=0.25)
        ax = fig.add_subplot(gs[0, :])
        ax.plot(x.index, x.values, color=MUTED, lw=0.4)
        for r in rr.itertuples():
            ax.axvspan(r.RUN_START, r.RUN_END + pd.Timedelta('30min'), color=COLOUR[r.TYPE], alpha=0.35, lw=0)
        ax.axhline(0, color=INK, lw=0.6)
        ax.grid(axis='y', color=GRID, lw=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.set_ylabel('MW (buses combined)')
        note = '  ·  site left out altogether' if s_name in EXCLUDE_SITES else ''
        ax.set_title(f"{s_name} ({site_buses[s_name]})  ·  {(rr.TYPE == 'zero_run').sum()} zero runs (orange), "
                     f"{(rr.TYPE == 'flat_run').sum()} flat runs (violet), {(rr.TYPE == 'night_reverse').sum()} night-reverse "
                     f"days (green)  ·  {rr['HOURS'].sum():,.1f} h excluded{note}",
                     loc='left', fontsize=9.5, color=INK)
        for k, r in enumerate(rr.head(6).itertuples()):
            a = fig.add_subplot(gs[1 + k // 3, k % 3])
            pad = pd.Timedelta(hours=max(6, min(r.HOURS * 0.5, 96)))
            w = x.loc[r.RUN_START - pad:r.RUN_END + pad]
            a.plot(w.index, w.values, color=MUTED, lw=0.8)
            a.axvspan(r.RUN_START, r.RUN_END + pd.Timedelta('30min'), color=COLOUR[r.TYPE], alpha=0.35, lw=0)
            a.axhline(0, color=INK, lw=0.5)
            a.grid(axis='y', color=GRID, lw=0.5)
            a.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=4))
            a.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            what = ('zero' if r.TYPE == 'zero_run' else f'flat at {r.VALUE_MW:.2f} MW' if r.TYPE == 'flat_run'
                    else f'reverse at night, min {r.VALUE_MW:.1f} MW')
            a.set_title(f"{r.RUN_START:%d %b %Y %H:%M}: {what}, {r.HOURS:g} h", loc='left', fontsize=7.5, color=INK)
        if len(rr) > 6:
            fig.text(0.01, 0.01, f'{len(rr) - 6} shorter runs not zoomed; all runs are in load_quality_flags.csv.',
                     fontsize=7, color=MUTED)
        pdf.savefig(fig)
        plt.close(fig)

n_days = (site.shape[1] - len(EXCLUDE_SITES)) * len(set(full_index.normalize()))
print(f"{len(series)} buses with data, {int((summary['ROWS'] == 0).sum())} empty file(s); "
      f"blank values {int(summary['BLANK'].sum())}, absent timestamps {int(summary['ABSENT'].sum())}, "
      f"gaps in load_missing_intervals.csv {len(gaps)}")
print(f"Appendix plots: {len(order)} sites -> excluded_runs_plots.pdf")
print(flags.groupby('TYPE').agg(runs=('HOURS', 'size'), hours=('HOURS', 'sum'), sites=('SITE', 'nunique')).to_string())
print(f"Site-days left out: {len(excluded_days)} at {excluded_days['SITE'].nunique()} sites "
      f"({len(excluded_days) / n_days:.2%}); sites left out: {', '.join(EXCLUDE_SITES)}")

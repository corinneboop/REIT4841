### STEP 2 - Compare AEMO measured actuals with satellite actuals (thesis plan step 2c).

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

START_DATE, END_DATE = '2025-07-01', '2026-06-30'
MEASURED_COLOUR, SATELLITE_COLOUR = '#2a78d6', '#eda100'
DAYLIGHT_MW = 1.0
DT = 0.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP1_OUT = os.path.join(SCRIPT_DIR, '..', '1_unzip_and_join', 'output')
MEAS_FILE = os.path.join(STEP1_OUT, 'QLDSOUTH_measured_final.csv')
SAT_FILE = os.path.join(STEP1_OUT, 'QLDSOUTH_satellite_final.csv')
REMOVED_FILE = os.path.join(STEP1_OUT, 'removed_days.csv')
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

meas = pd.read_csv(MEAS_FILE, index_col='INTERVAL_START', parse_dates=True)['POWER_MW']
sat = pd.read_csv(SAT_FILE, index_col='INTERVAL_START', parse_dates=True)['POWER_MW']
df = pd.concat({'meas': meas, 'sat': sat}, axis=1, join='inner')
df['start'] = df.index
if START_DATE:
    df = df[df['start'] >= pd.Timestamp(START_DATE)]
if END_DATE:
    df = df[df['start'] < pd.Timestamp(END_DATE) + pd.Timedelta('1D')]
df = df.copy()
df['date'] = df['start'].dt.date
df['month'] = df['start'].dt.month
df['hour'] = df['start'].dt.hour + df['start'].dt.minute / 60 + 0.25

n = df.groupby('date').size()
short = n[n < 48]
df = df[~df['date'].isin(set(short.index))]
removed = pd.read_csv(REMOVED_FILE, parse_dates=['DATE'])
if START_DATE:
    removed = removed[removed['DATE'] >= pd.Timestamp(START_DATE)]
if END_DATE:
    removed = removed[removed['DATE'] <= pd.Timestamp(END_DATE)]
print(f"{len(removed)} day(s) removed in step 1b (removed_days.csv); {len(short)} incomplete day(s) dropped here; "
      f"comparing {df['date'].nunique()} complete days")
print(removed.to_string(index=False))

daily = df.groupby('date').agg(meas=('meas', lambda s: s.sum() * DT / 1000),
                               sat=('sat', lambda s: s.sum() * DT / 1000), month=('month', 'first'))
day = df[(df['meas'] > DAYLIGHT_MW) | (df['sat'] > DAYLIGHT_MW)]

def metrics(m, s, level, unit):
    diff = s - m
    return {'Level': level, 'Unit': unit, 'N': len(m),
            'Measured mean': m.mean(), 'Satellite mean': s.mean(),
            'Bias (sat - meas)': diff.mean(), 'Bias %': 100 * diff.mean() / m.mean(),
            'MAE': diff.abs().mean(), 'RMSE': np.sqrt((diff ** 2).mean()),
            'nRMSE % (of measured mean)': 100 * np.sqrt((diff ** 2).mean()) / m.mean(),
            'Pearson r': np.corrcoef(m, s)[0, 1], 'R2': np.corrcoef(m, s)[0, 1] ** 2}

table = pd.DataFrame([metrics(day['meas'], day['sat'], 'Half-hourly (daylight)', 'MW'),
                      metrics(daily['meas'], daily['sat'], 'Daily energy', 'GWh')])
table.round(3).to_csv(os.path.join(OUTPUT_FOLDER, 'comparison_metrics.csv'), index=False)
print('\n' + table.round(3).to_string(index=False))

monthly = daily.groupby('month').agg(days=('meas', 'size'), measured_median_GWh=('meas', 'median'),
                                     satellite_median_GWh=('sat', 'median'))
monthly['median_diff_%'] = 100 * ((daily['sat'] - daily['meas']) / daily['meas']).groupby(daily['month']).median()
monthly.index = [pd.Timestamp(2000, m, 1).strftime('%b') for m in monthly.index]
monthly.index.name = 'Month'
monthly.round(2).to_csv(os.path.join(OUTPUT_FOLDER, 'monthly_comparison.csv'))

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
style = lambda ax: (ax.grid(color='#e0e0e0', linewidth=0.6), ax.set_axisbelow(True),
                    ax.spines[['top', 'right']].set_visible(False))

ax = axes[0, 0]
hb = ax.hexbin(day['meas'], day['sat'], gridsize=60, cmap='Blues', mincnt=1, bins='log')
lim = max(day['meas'].max(), day['sat'].max()) * 1.03
ax.plot([0, lim], [0, lim], color='#444444', lw=1, ls='--', label='1:1 line')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('Measured (MW)'); ax.set_ylabel('Satellite (MW)')
ax.set_title('(a) Half-hourly power, daylight intervals', loc='left', fontsize=11)
ax.legend(loc='upper left', frameon=False)
fig.colorbar(hb, ax=ax, label='Number of half-hours (log)')
style(ax)

ax = axes[0, 1]
ax.scatter(daily['meas'], daily['sat'], s=22, color=MEASURED_COLOUR, alpha=0.8, edgecolor='white', linewidth=0.5,
           label='Day')
lim = max(daily['meas'].max(), daily['sat'].max()) * 1.05
ax.plot([0, lim], [0, lim], color='#444444', lw=1, ls='--', label='1:1 line')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('Measured daily energy (GWh)'); ax.set_ylabel('Satellite daily energy (GWh)')
ax.set_title('(b) Daily energy', loc='left', fontsize=11)
ax.legend(loc='upper left', frameon=False)
style(ax)

ax = axes[1, 0]
df['diff'] = df['sat'] - df['meas']
g = df.groupby('hour')['diff']
ax.fill_between(g.median().index, g.quantile(.25), g.quantile(.75), color=MEASURED_COLOUR, alpha=0.25, lw=0,
                label='P25-P75')
ax.plot(g.median().index, g.median(), color=MEASURED_COLOUR, lw=2, label='Median')
ax.axhline(0, color='#444444', lw=1)
ax.set_xlim(4, 20); ax.set_xticks(range(4, 21, 2))
ax.set_xticklabels([f'{h:02d}:00' for h in range(4, 21, 2)])
ax.set_xlabel('Time of day (AEST)'); ax.set_ylabel('Satellite - measured (MW)')
ax.set_title('(c) Difference by time of day', loc='left', fontsize=11)
ax.legend(loc='lower left', frameon=False, fontsize=9)
style(ax)

ax = axes[1, 1]
x = np.arange(len(monthly))
w = 0.38
ax.bar(x - w / 2 - 0.01, monthly['measured_median_GWh'], w, color=MEASURED_COLOUR, label='Measured')
ax.bar(x + w / 2 + 0.01, monthly['satellite_median_GWh'], w, color=SATELLITE_COLOUR, label='Satellite')
for i, pct in enumerate(monthly['median_diff_%']):
    top = max(monthly['measured_median_GWh'].iloc[i], monthly['satellite_median_GWh'].iloc[i])
    ax.text(i, top + 0.4, f'{pct:+.0f}%', ha='center', fontsize=8, color='#555555')
ax.set_xticks(x); ax.set_xticklabels(monthly.index)
ax.set_ylabel('Median daily energy (GWh)')
ax.set_title('(d) Monthly median daily energy (label = median satellite vs measured %)', loc='left', fontsize=11)
ax.legend(loc='upper center', ncol=2, frameon=False)
style(ax); ax.grid(axis='x', visible=False)

fig.suptitle('QLDSOUTH rooftop PV - AEMO measured vs satellite actuals', x=0.01, ha='left', fontsize=13)
fig.text(0.01, 0.945, f'{df["date"].nunique()} complete days, {START_DATE} to {END_DATE}; '
         f'{len(removed)} day(s) with any blank/absent interval in either source removed.', fontsize=9, color='#555555')
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUTPUT_FOLDER, 'measured_vs_satellite.png'), dpi=200)
print('Saved to', OUTPUT_FOLDER)

### STEP 4 - Monthly diurnal (time-of-day) profiles of rooftop PV for QLDSOUTH (AEMO measured actuals, final dataset).

import os
import pandas as pd
import matplotlib.pyplot as plt

START_DATE, END_DATE = '2025-07-01', '2026-06-30'
MONTH_ORDER = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
INNER_BAND, OUTER_BAND = (25, 75), (10, 90)
MEDIAN_COLOUR, INNER_COLOUR, OUTER_COLOUR = '#e34948', '#2a78d6', '#eda100'
HOUR_RANGE = (4, 20)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, '..', '1_unzip_and_join', 'output', 'QLDSOUTH_measured_final.csv')
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'output')
BY_MONTH = os.path.join(OUTPUT_FOLDER, 'by_month')
os.makedirs(BY_MONTH, exist_ok=True)
DATE_TEXT = f"{pd.Timestamp(START_DATE):%d/%m/%Y} to {pd.Timestamp(END_DATE):%d/%m/%Y}"
NOTE = ('Each line/band is the statistic of that half-hour across all days in the month; '
        'it is not the output of a single real day.')

df = pd.read_csv(INPUT_FILE, index_col='INTERVAL_START', parse_dates=True)
df['start'] = df.index
df = df[(df['start'] >= pd.Timestamp(START_DATE)) & (df['start'] < pd.Timestamp(END_DATE) + pd.Timedelta('1D'))]
df['date'] = df['start'].dt.normalize()
n = df.groupby('date')['POWER_MW'].count()
df = df[df['date'].isin(n.index[n == 48])]
df['month'] = df['start'].dt.month
df['hour'] = df['start'].dt.hour + df['start'].dt.minute / 60 + 0.25

pcts = sorted(set(INNER_BAND + OUTER_BAND))
profiles, days = {}, {}
for m in MONTH_ORDER:
    g = df[df['month'] == m].groupby('hour')['POWER_MW']
    p = pd.DataFrame({'median': g.median()})
    for q in pcts:
        p[f'P{q}'] = g.quantile(q / 100)
    profiles[m] = p
    days[m] = df.loc[df['month'] == m, 'date'].nunique()
table = pd.concat({pd.Timestamp(2000, m, 1).strftime('%b'): profiles[m] for m in MONTH_ORDER}, names=['Month', 'Hour'])
table.round(1).to_csv(os.path.join(OUTPUT_FOLDER, 'diurnal_profile_monthly.csv'))
ymax = max(p[f'P{OUTER_BAND[1]}'].max() for p in profiles.values()) * 1.05
print(f"{df['date'].nunique()} complete days, {DATE_TEXT}. Median peak (MW) by month:")
print({pd.Timestamp(2000, m, 1).strftime('%b'): round(profiles[m]['median'].max()) for m in MONTH_ORDER})

def draw(ax, m, legend=False, labels=True):
    p = profiles[m]
    x = p.index
    lo_o, hi_o, lo_i, hi_i = (p[f'P{OUTER_BAND[0]}'], p[f'P{OUTER_BAND[1]}'],
                              p[f'P{INNER_BAND[0]}'], p[f'P{INNER_BAND[1]}'])
    ax.fill_between(x, lo_o, lo_i, color=OUTER_COLOUR, alpha=0.35, lw=0, label=f'P{OUTER_BAND[0]}-P{OUTER_BAND[1]}')
    ax.fill_between(x, hi_i, hi_o, color=OUTER_COLOUR, alpha=0.35, lw=0)
    ax.fill_between(x, lo_i, hi_i, color=INNER_COLOUR, alpha=0.35, lw=0, label=f'P{INNER_BAND[0]}-P{INNER_BAND[1]}')
    ax.plot(x, p['median'], color=MEDIAN_COLOUR, lw=2, label='Median')
    ax.set_title(f"{pd.Timestamp(2000, m, 1):%B} (n = {days[m]} days)", loc='left', fontsize=10)
    ax.set_xlim(*HOUR_RANGE)
    ax.set_ylim(0, ymax)
    ticks = range(HOUR_RANGE[0], HOUR_RANGE[1] + 1, 4)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{h:02d}:00' for h in ticks] if labels else [])
    ax.grid(color='#e0e0e0', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    if legend:
        ax.legend(loc='upper left', frameon=False, fontsize=8)

fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharey=True)
for i, (ax, m) in enumerate(zip(axes.flat, MONTH_ORDER)):
    draw(ax, m, legend=(i == 0), labels=(i >= 8))
for ax in axes[:, 0]:
    ax.set_ylabel('Rooftop PV output (MW)')
for ax in axes[-1, :]:
    ax.set_xlabel('Time of day (AEST)')
fig.suptitle('QLDSOUTH rooftop PV diurnal profile by month - AEMO measured actuals', x=0.01, ha='left', fontsize=13)
fig.text(0.01, 0.945, f'{NOTE} {DATE_TEXT}.', fontsize=9, color='#555555')
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUTPUT_FOLDER, 'diurnal_profile_monthly.png'), dpi=200)
plt.close(fig)

for m in MONTH_ORDER:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    draw(ax, m, legend=True)
    ax.set_xticks(range(HOUR_RANGE[0], HOUR_RANGE[1] + 1, 2))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(HOUR_RANGE[0], HOUR_RANGE[1] + 1, 2)])
    ax.set_xlabel('Time of day (AEST)')
    ax.set_ylabel('Rooftop PV output (MW)')
    fig.suptitle(f"QLDSOUTH rooftop PV diurnal profile - {pd.Timestamp(2000, m, 1):%B} - AEMO measured actuals",
                 x=0.01, ha='left', fontsize=12)
    fig.text(0.01, 0.9, 'Each line/band is the statistic of that half-hour across all days in the month;\n'
             f'it is not the output of a single real day. {DATE_TEXT}.', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(BY_MONTH, f"diurnal_{m:02d}_{pd.Timestamp(2000, m, 1):%b}.png"), dpi=200)
    plt.close(fig)
print('Saved to', OUTPUT_FOLDER)

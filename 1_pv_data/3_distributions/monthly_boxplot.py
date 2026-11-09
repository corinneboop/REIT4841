### STEP 3 - Monthly boxplots of rooftop PV for the whole QLDSOUTH area (AEMO measured actuals, final dataset;
### area totals, not normalised by installed capacity):
###   daily total energy (GWh) - how much solar the area generates in a day
###   daily peak output (GW)   - the highest half-hour output in the area that day

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats

START_DATE, END_DATE = '2025-07-01', '2026-06-30'
MONTH_ORDER = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
BOX_COLOUR = '#2a78d6'
WHISKERS = 1.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, '..', '1_unzip_and_join', 'output', 'QLDSOUTH_measured_final.csv')
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
DATE_TEXT = f"{pd.Timestamp(START_DATE):%d/%m/%Y} to {pd.Timestamp(END_DATE):%d/%m/%Y}"

df = pd.read_csv(INPUT_FILE, index_col='INTERVAL_START', parse_dates=True)
df['start'] = df.index
df = df[(df['start'] >= pd.Timestamp(START_DATE)) & (df['start'] < pd.Timestamp(END_DATE) + pd.Timedelta('1D'))]
daily = df.groupby(df['start'].dt.normalize()).agg(
    energy_GWh=('POWER_MW', lambda s: s.sum() * 0.5 / 1000),
    peak_GW=('POWER_MW', lambda s: s.max() / 1000),
    n_intervals=('POWER_MW', 'count'))
daily = daily[daily['n_intervals'] == 48]
daily['month'] = daily.index.month
print(f"{len(daily)} complete days, {DATE_TEXT}")

MEASURES = [('energy_GWh', 'Daily energy', 'GWh'), ('peak_GW', 'Daily peak', 'GW')]
rows = []
for col, name, unit in MEASURES:
    for m in MONTH_ORDER:
        s = daily.loc[daily['month'] == m, col]
        b = boxplot_stats(s.values, whis=WHISKERS)[0]
        low, high = s[s < b['whislo']], s[s > b['whishi']]
        rows.append({'Measure': f'{name} ({unit})', 'Month': pd.Timestamp(2000, m, 1).strftime('%b'), 'Days': len(s),
                     'Min': s.min(), 'Lower whisker': b['whislo'], 'P25': b['q1'], 'Median': b['med'],
                     'P75': b['q3'], 'Upper whisker': b['whishi'], 'Max': s.max(), 'IQR': b['iqr'], 'Mean': s.mean(),
                     'Outliers low (n)': len(low), 'Outliers high (n)': len(high),
                     'Outlier days': '; '.join(f'{d:%d/%m/%Y} ({v:.2f})' for d, v in pd.concat([low, high]).items())})
summary = pd.DataFrame(rows).round(2)
summary.to_csv(os.path.join(OUTPUT_FOLDER, 'monthly_summary_measured.csv'), index=False)
print(summary.to_string(index=False))

def boxplot_figure(col, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot([daily.loc[daily['month'] == m, col].values for m in MONTH_ORDER], patch_artist=True,
               widths=0.6, whis=WHISKERS,
               boxprops=dict(facecolor=BOX_COLOUR, alpha=0.75, edgecolor='#444444'),
               medianprops=dict(color='#222222', linewidth=2), whiskerprops=dict(color='#555555'),
               capprops=dict(color='#555555'),
               flierprops=dict(marker='o', markersize=4, markerfacecolor='none', markeredgecolor='#777777'))
    n_days = [int((daily['month'] == m).sum()) for m in MONTH_ORDER]
    ax.set_xticks(range(1, len(MONTH_ORDER) + 1))
    ax.set_xticklabels([f"{pd.Timestamp(2000, m, 1):%b}\nn = {n}" for m, n in zip(MONTH_ORDER, n_days)])
    ax.set_xlabel('Month (n = number of days)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', color='#e0e0e0', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    whisker_text = '1.5 x IQR' if WHISKERS == 1.5 else f'P{WHISKERS[0]}-P{WHISKERS[1]}'
    fig.suptitle(title, x=0.01, ha='left', fontsize=13)
    fig.text(0.01, 0.905, f'Box = P25-P75, line = median, whiskers = {whisker_text}, circles = days beyond the whiskers; '
             f'{len(daily)} days, {DATE_TEXT}.', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUTPUT_FOLDER, filename), dpi=200)
    plt.close(fig)


boxplot_figure('energy_GWh', 'Daily total rooftop PV energy (GWh)',
               'QLDSOUTH daily rooftop PV energy by month - AEMO measured actuals', 'monthly_boxplot_measured.png')
boxplot_figure('peak_GW', 'Daily peak rooftop PV output (GW)',
               'QLDSOUTH daily peak rooftop PV output by month - AEMO measured actuals', 'monthly_boxplot_peak.png')
print('Saved to', OUTPUT_FOLDER)

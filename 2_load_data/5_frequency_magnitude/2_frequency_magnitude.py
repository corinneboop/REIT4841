### Step 5.2 - Frequency vs magnitude of reverse power flow in each postcode, relative to hosting capacity.
### Run 1_network_capacity.py (same folder) first.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.titlesize': 12,
                     'legend.fontsize': 11})

MIN_DAYS = 10
PCTS = (75, 90)
MAIN_PCT = 90
FREQ_CUTS = (10, 50)
MAG_CUTS = (50, 100)
FREQ_CLASSES = ['rare', 'occasional', 'frequent']
MAG_CLASSES = ['low', 'approaching', 'above']
DOT_COLOUR = '#2a78d6'

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD_DIR = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
S4 = os.path.join(LOAD_DIR, '4_reverse_flow', 'output')

d4 = pd.read_csv(os.path.join(S4, 'reverse_flow_daily_by_postcode.csv'), dtype={'POSTCODE': str}, parse_dates=['DATE'])
d4 = d4[~d4['DAY_FLAGGED']]
d5 = pd.read_csv(os.path.join(OUT, 'capacity_daily_by_postcode.csv'), dtype={'POSTCODE': str}, parse_dates=['DATE'])
ev = d4.merge(d5[['DATE', 'POSTCODE', 'PEAK_PCT_OF_CEG_main']], on=['DATE', 'POSTCODE'], how='left')
ev['REV'] = ev['REVERSE_TRAP_MWh'] > 0
cp = pd.read_csv(os.path.join(OUT, 'capacity_by_postcode.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')

g = ev.groupby('POSTCODE')
t = pd.DataFrame({'DAYS_USED': g.size(), 'DAYS_WITH_REVERSE': g['REV'].sum()})
t['PCT_DAYS_WITH_REVERSE'] = 100 * t['DAYS_WITH_REVERSE'] / t['DAYS_USED']
t['ANNUAL_REVERSE_MWh'] = g['REVERSE_TRAP_MWh'].sum() * 365 / t['DAYS_USED']
r = ev[ev['REV']].groupby('POSTCODE')
for pc in PCTS:
    t[f'P{pc}_PCT_CEG'] = r['PEAK_PCT_OF_CEG_main'].quantile(pc / 100)
no_cap = cp['STATUS'].reindex(t.index).eq('no published capacity')
t.loc[no_cap, [f'P{pc}_PCT_CEG' for pc in PCTS]] = np.nan
t = t.join(cp[['STATUS_DETAIL', 'MAIN_REVERSE_SITE']])
t['ANALYSED'] = t['DAYS_WITH_REVERSE'] >= MIN_DAYS
p = t[t['ANALYSED']].copy()


def classify(x, cuts, labels):
    return pd.cut(x, [-np.inf, *cuts, np.inf], right=False, labels=labels)


p['FREQ_CLASS'] = classify(p['PCT_DAYS_WITH_REVERSE'], FREQ_CUTS, FREQ_CLASSES)
p['MAG_CLASS'] = classify(p[f'P{MAIN_PCT}_PCT_CEG'], MAG_CUTS, MAG_CLASSES)
for c in ('FREQ_CLASS', 'MAG_CLASS'):
    t[c] = p[c].astype(object).reindex(t.index)
t.sort_values(f'P{MAIN_PCT}_PCT_CEG', ascending=False).round(3).to_csv(
    os.path.join(OUT, 'frequency_magnitude_by_postcode.csv'), index_label='POSTCODE')
print(f"{len(p)} postcodes with >= {MIN_DAYS} reverse-flow days ({len(t) - len(p)} left out); "
      f"{p['MAG_CLASS'].isna().sum()} of them have no published capacity")
print(f"Median postcode: reverse flow on {p['PCT_DAYS_WITH_REVERSE'].median():.1f}% of days; "
      + ', '.join(f"P{pc} peak {p[f'P{pc}_PCT_CEG'].median():.1f}% of CEG" for pc in PCTS))
for pc in PCTS:
    print(f"P{pc} at or above 100% of CEG: {', '.join(p.index[p[f'P{pc}_PCT_CEG'] >= MAG_CUTS[1]])}")

cs = (p.dropna(subset=['MAG_CLASS']).groupby(['FREQ_CLASS', 'MAG_CLASS'], observed=False)
      .agg(POSTCODES=('ANNUAL_REVERSE_MWh', 'size'), REVERSE_GWh=('ANNUAL_REVERSE_MWh', lambda s: s.sum() / 1000),
           LIST=('ANNUAL_REVERSE_MWh', lambda s: ', '.join(s.sort_values(ascending=False).index[:10]))))
cs['PCT_OF_REVERSE_ENERGY'] = 100 * cs['REVERSE_GWh'] / cs['REVERSE_GWh'].sum()
cs.round(2).to_csv(os.path.join(OUT, 'class_summary.csv'))
print(f'\nClasses (frequency: % days cut at {FREQ_CUTS}; magnitude: P{MAIN_PCT} peak % of CEG cut at {MAG_CUTS}):')
print(cs.drop(columns='LIST').round(1).to_string())

size = lambda mwh: 10 + 300 * np.sqrt(mwh / p['ANNUAL_REVERSE_MWh'].max())
y = f'P{MAIN_PCT}_PCT_CEG'
d = p[p[y] > 0]
fig, ax = plt.subplots(figsize=(11.5, 7.8))
ax.scatter(d['PCT_DAYS_WITH_REVERSE'], d[y], s=size(d['ANNUAL_REVERSE_MWh']), color=DOT_COLOUR, alpha=0.75,
           edgecolor='white', linewidth=0.6, zorder=3)
ax.set_yscale('log'); ax.set_xlim(0, 100)
ylo, yhi = d[y].min() / 2, d[y].max() * 3
ax.set_ylim(ylo, yhi)
for v in FREQ_CUTS:
    ax.axvline(v, color='#777777', ls='--', lw=1, zorder=1)
ax.axhline(MAG_CUTS[0], color='#777777', ls='--', lw=1, zorder=1)
ax.axhline(MAG_CUTS[1], color='#c0392b', lw=1.2, zorder=1)
xc = {'rare': FREQ_CUTS[0] / 2, 'occasional': sum(FREQ_CUTS) / 2, 'frequent': (FREQ_CUTS[1] + 100) / 2}
xl = {'rare': 0.8, 'occasional': FREQ_CUTS[0] + 0.8, 'frequent': FREQ_CUTS[1] + 0.8}
yc = {'low': ylo * 1.35, 'approaching': np.sqrt(MAG_CUTS[0] * MAG_CUTS[1]), 'above': MAG_CUTS[1] * 1.7}

fa = d[(d['FREQ_CLASS'] == 'frequent') & (d['MAG_CLASS'] == 'above')].sort_values(y, ascending=False)
ys = np.geomspace(yhi / 1.6, MAG_CUTS[1] * 1.25, max(len(fa), 2))
for (pcode, r_), yt in zip(fa.iterrows(), ys):
    ax.annotate(f"{pcode} ({r_['MAIN_REVERSE_SITE']})", (r_['PCT_DAYS_WITH_REVERSE'], r_[y]), xytext=(70, yt),
                textcoords='data', fontsize=10, color='#333333', va='center', ha='right', zorder=5,
                arrowprops=dict(arrowstyle='-', color='#999999', lw=0.6, shrinkA=2, shrinkB=4))
for lab, xv in xc.items():
    ax.text(xv, yhi / 1.1, lab.upper(), ha='center', va='top', fontsize=11.5, color='#555555')
ax.set_xlabel('Frequency: % of days with reverse power flow')
ax.set_ylabel(f'Magnitude: P{MAIN_PCT} daily peak reverse flow, % of capacity for embedded generation (log)')
ax.set_title(f'How often and how hard postcode reverse power flow pushes against hosting capacity - {len(d)} postcodes\n'
             f'Frequency: rare < {FREQ_CUTS[0]}%, occasional, frequent >= {FREQ_CUTS[1]}% of days. '
             f'Magnitude: low < {MAG_CUTS[0]}%, approaching, above >= {MAG_CUTS[1]}% of CEG (DAPR 2025/26).',
             loc='left', fontsize=13.5)
ax.grid(color='#e6e6e6', linewidth=0.6, which='major'); ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
h = [plt.scatter([], [], s=size(v), color=DOT_COLOUR, alpha=0.75, edgecolor='white', label=f'{v / 1000:g} GWh/yr')
     for v in (500, 2000, 8000)]
fig.legend(handles=h, loc='lower center', ncol=len(h), frameon=False, fontsize=11,
           title='Dot size = annual reverse energy', title_fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(OUT, 'fig_frequency_magnitude.png'), dpi=200)
plt.close(fig)
print('Saved to', OUT)

### Step 6.3 - Frequency vs magnitude of reverse power flow (5_frequency_magnitude), coloured by battery alignment.
### Run 5_frequency_magnitude and 1_battery_alignment.py (same folder) first.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.titlesize': 12,
                     'legend.fontsize': 11})

MAIN_PCT = 90
FREQ_CUTS = (10, 50)
MAG_CUTS = (50, 100)
FREQ_CLASSES = ['rare', 'occasional', 'frequent']
MAG_CLASSES = ['low', 'approaching', 'above']
ALIGN_COLOURS = {'under-aligned': '#e34948', 'aligned': '#2a78d6', 'over-aligned': '#eda100',
                 'little/no need': '#9a9a9a'}

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD_DIR = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
S5 = os.path.join(LOAD_DIR, '5_frequency_magnitude', 'output')

fm = pd.read_csv(os.path.join(S5, 'frequency_magnitude_by_postcode.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
al = pd.read_csv(os.path.join(OUT, 'battery_alignment_by_postcode.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')
p = fm[fm['ANALYSED']].join(al[['ALIGNMENT_P90', 'COVERAGE_P90']])
p['ALIGNMENT'] = p['ALIGNMENT_P90'].str.replace(r' \(.*\)', '', regex=True)
y = f'P{MAIN_PCT}_PCT_CEG'
d = p[p[y] > 0].copy()
d['FREQ_CLASS'] = pd.Categorical(d['FREQ_CLASS'], FREQ_CLASSES)
d['MAG_CLASS'] = pd.Categorical(d['MAG_CLASS'], MAG_CLASSES)

cs = d.groupby(['FREQ_CLASS', 'MAG_CLASS'], observed=False).agg(
    POSTCODES=('ANNUAL_REVERSE_MWh', 'size'), REVERSE_GWh=('ANNUAL_REVERSE_MWh', lambda s: s.sum() / 1000))
cs['PCT_OF_REVERSE_ENERGY'] = 100 * cs['REVERSE_GWh'] / cs['REVERSE_GWh'].sum()
al_counts = d.groupby(['FREQ_CLASS', 'MAG_CLASS', 'ALIGNMENT'], observed=False).size().unstack(fill_value=0)
cs = cs.join(al_counts.reindex(columns=list(ALIGN_COLOURS), fill_value=0).rename(columns=str.upper))
cs.round(2).to_csv(os.path.join(OUT, 'class_alignment_summary.csv'))
print(f'{len(d)} postcodes; frequency x magnitude class by P{MAIN_PCT} battery alignment:')
print(cs.round(1).to_string())

size = lambda mwh: 10 + 300 * np.sqrt(mwh / p['ANNUAL_REVERSE_MWh'].max())
fig, ax = plt.subplots(figsize=(11.5, 7.8))
for c, colour in ALIGN_COLOURS.items():
    s = d[d['ALIGNMENT'] == c]
    ax.scatter(s['PCT_DAYS_WITH_REVERSE'], s[y], s=size(s['ANNUAL_REVERSE_MWh']), color=colour, alpha=0.75,
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
for (fcl, mcl), r_ in cs.iterrows():
    if r_['POSTCODES'] == 0:
        continue
    ax.text(xl[fcl], yc[mcl], f"{int(r_['POSTCODES'])} postcode{'s' if r_['POSTCODES'] != 1 else ''}, "
            f"{int(r_['UNDER-ALIGNED'])} under-aligned\n{r_['PCT_OF_REVERSE_ENERGY']:.0f}% of energy",
            ha='left', va='center', fontsize=11, color='#333333', zorder=4,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85),
            fontweight='bold' if (fcl, mcl) == ('frequent', 'above') else 'normal')
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
ax.set_title(f'Reverse power flow against hosting capacity, by battery alignment - {len(d)} postcodes\n'
             f'Frequency: rare < {FREQ_CUTS[0]}%, occasional, frequent >= {FREQ_CUTS[1]}% of days. '
             f'Magnitude: low < {MAG_CUTS[0]}%, approaching, above >= {MAG_CUTS[1]}% of CEG (DAPR 2025/26).',
             loc='left', fontsize=13.5)
ax.grid(color='#e6e6e6', linewidth=0.6, which='major'); ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
h = [Line2D([], [], marker='o', ls='', markersize=8, markerfacecolor=c, markeredgecolor='white',
            label=f"{k} ({(d['ALIGNMENT'] == k).sum()})") for k, c in ALIGN_COLOURS.items() if (d['ALIGNMENT'] == k).any()]
h += [plt.scatter([], [], s=size(v), color='#bbbbbb', edgecolor='white', label=f'{v / 1000:g} GWh/yr')
      for v in (500, 2000, 8000)]
fig.legend(handles=h, loc='lower center', ncol=len(h), frameon=False, fontsize=11,
           title='Battery alignment (P90 coverage: under < 0.5, over > 2)  |  dot size = annual reverse energy',
           title_fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(OUT, 'fig_frequency_magnitude_alignment.png'), dpi=200)
plt.close(fig)
print('Saved to', OUT)

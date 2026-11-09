### STEP 7b - Technical alignment: is battery storage in each postcode under-, well- or over-aligned with its need?
### (Technical data only; no social variable is used.)

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
SOCIAL_DIR = os.path.dirname(HERE)
CODE = os.path.dirname(SOCIAL_DIR)
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
POA_SHP = os.path.join(CODE, '2_load_data', 'data', 'POA_2021_AUST_GDA2020_SHP', 'POA_2021_AUST_GDA2020.shp')
UNDER, OVER = 0.5, 2.0
CONSTRAINED = ['above capacity', 'near capacity']
STORAGE = 'battery_kWh_per_solar_site_all'
ALIGN = ['under-aligned (above capacity)', 'under-aligned (below capacity)', 'aligned', 'over-aligned']
COLOURS = dict(zip(ALIGN, ['#99000d', '#ef3b2c', '#fc9272', '#fee5d9']))

d = pd.read_csv(os.path.join(SOCIAL_DIR, '2_analysis_dataset', 'output', 'analysis_dataset.csv'), dtype={'POSTCODE': str}).set_index('POSTCODE')


ABOVE = d['network_status'] == 'above capacity'


def align(need, under=UNDER, over=OVER):
    cov = (d[STORAGE] / need).where(need > 0)
    lab = np.select([cov.isna() | (cov > over), (cov < under) & ABOVE, cov < under],
                    ['over-aligned', 'under-aligned (above capacity)', 'under-aligned (below capacity)'], 'aligned')
    return pd.Series(lab, index=d.index), cov


need = d['need_p90_kWh_per_site']
al, cov = align(need)
gap = (need - d[STORAGE]).clip(lower=0)
t = pd.DataFrame({'region': d['region'], 'main_reverse_site': d['main_reverse_site'], 'solar_sites': d['solar_sites'],
                  'population': d['population'], 'need_p90_kWh_per_site': need, 'storage_kWh_per_site': d[STORAGE],
                  'coverage_p90': cov, 'gap_kWh_per_site': gap, 'gap_MWh': gap * d['solar_sites'] / 1000,
                  'network_status': d['network_status'], 'constrained': d['network_status'].isin(CONSTRAINED),
                  'alignment': al})

SPECS = list(itertools.product([('P90', 'need_p90_kWh_per_site'), ('P75', 'need_p75_kWh_per_site')], [0.25, 0.5, 0.75], [1.5, 2.0, 3.0]))
alt, rows = {}, []
for (nl, nc), u, o in SPECS:
    a, _ = align(d[nc], u, o)
    alt[(nl, u, o)] = a
    rows.append({'need': nl, 'under_cut': u, 'over_cut': o, **{k: int((a == k).sum()) for k in ALIGN},
                 'agree_with_main': (a == al).mean(), 'main': (nl, u, o) == ('P90', UNDER, OVER)})
sens = pd.DataFrame(rows)
sens.round(3).to_csv(os.path.join(OUT, '7b_sensitivity.csv'), index=False)
A = pd.DataFrame(alt)
t['agreement'] = A.eq(al, axis=0).mean(axis=1)
t['robustly_under_aligned'] = A.isin(ALIGN[:2]).mean(axis=1) >= 0.75
t.round(4).to_csv(os.path.join(OUT, '7b_alignment.csv'), index_label='POSTCODE')

summ = t.groupby('alignment').agg(postcodes=('alignment', 'size'), constrained=('constrained', 'sum'),
                                  solar_sites=('solar_sites', 'sum'), people=('population', 'sum'),
                                  median_need_kWh=('need_p90_kWh_per_site', 'median'),
                                  median_storage_kWh=('storage_kWh_per_site', 'median'),
                                  median_coverage=('coverage_p90', 'median'), total_gap_MWh=('gap_MWh', 'sum'),
                                  median_agreement=('agreement', 'median')).reindex(ALIGN)
summ['share_of_solar_sites'] = summ['solar_sites'] / summ['solar_sites'].sum()
summ['share_of_total_gap'] = summ['total_gap_MWh'] / summ['total_gap_MWh'].sum()
summ.round(3).to_csv(os.path.join(OUT, '7b_alignment_summary.csv'))
pd.set_option('display.width', 250)
print('ALIGNMENT'); print(summ.round(2).to_string())
print(f"\nRobustly under-aligned (under-aligned in >= 75% of {len(SPECS)} versions): {int(t['robustly_under_aligned'].sum())}")
print('\nSENSITIVITY'); print(sens.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 6.2))
for k in ALIGN[::-1]:
    q = t[t['alignment'] == k]
    ax.scatter(q['need_p90_kWh_per_site'], q['storage_kWh_per_site'], s=12 + q['solar_sites'] / 150, color=COLOURS[k],
               edgecolor='#666666' if k == 'over-aligned' else 'white', lw=0.5, alpha=0.9, label=f'{k} ({len(q)})', zorder=3)
x = np.linspace(0, t['need_p90_kWh_per_site'].max() * 1.05, 50)
ax.plot(x, UNDER * x, color='#333333', lw=0.9, ls='--', label=f'storage = {UNDER:g} x need')
ax.plot(x, OVER * x, color='#333333', lw=0.9, ls=':', label=f'storage = {OVER:g} x need')
ax.set_xlim(0, x[-1]); ax.set_ylim(0, max(t['storage_kWh_per_site'].max(), 3) * 1.05)
ax.set_xlabel('Need: P90 daily reverse flow per solar site (kWh)')
ax.set_ylabel('Installed battery storage per solar site (kWh)')
ax.set_title('Battery alignment with P90 need (dot size = solar sites)', loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=8, loc='upper right'); ax.grid(color='#eeeeee', lw=0.6); ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_7b_1_need_vs_storage.png'), dpi=180); plt.close(fig)

poa = gpd.read_file(POA_SHP, bbox=(151.8, -28.4, 153.7, -25.8)).rename(columns={'POA_CODE21': 'POSTCODE'})
gg = poa.merge(t[['alignment']].reset_index(), on='POSTCODE', how='inner')
fig, ax = plt.subplots(figsize=(7.5, 9))
poa[poa.intersects(gg.union_all().envelope)].plot(ax=ax, color='#f7f7f7', edgecolor='white', lw=0.2)
hand = []
for k in ALIGN:
    q = gg[gg['alignment'] == k]
    q.plot(ax=ax, color=COLOURS[k], edgecolor='white', lw=0.3)
    hand.append(plt.Rectangle((0, 0), 1, 1, color=COLOURS[k], label=f'{k} ({len(q)})'))
xmin, ymin, xmax, ymax = gg.total_bounds
ax.set_xlim(xmin - 0.05, xmax + 0.05); ax.set_ylim(ymin - 0.05, ymax + 0.05)
ax.legend(handles=hand, frameon=False, fontsize=8, loc='lower left')
ax.set_title(f'Battery alignment with P90 need (under < {UNDER:g} x need, over > {OVER:g} x need)', loc='left', fontsize=11); ax.set_axis_off()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_7b_2_alignment_map.png'), dpi=170); plt.close(fig)

LEVELS = [('Median day', 'need_median_kWh_per_site'), ('P75 day', 'need_p75_kWh_per_site'), ('P90 day (main)', 'need_p90_kWh_per_site')]
LCOL = {'under-aligned': '#b2182b', 'aligned': '#f4a582', 'over-aligned': '#2166ac', 'no need': '#bdbdbd'}
fig, axes = plt.subplots(1, 3, figsize=(16, 8))
lvl_rows = []
for ax, (lab, col) in zip(axes, LEVELS):
    cv = (d[STORAGE] / d[col]).where(d[col] > 0)
    a3 = pd.Series(np.select([cv.isna(), cv < UNDER, cv > OVER], ['no need', 'under-aligned', 'over-aligned'], 'aligned'), index=d.index)
    gl = poa.merge(a3.rename('a').reset_index(), on='POSTCODE', how='inner')
    poa[poa.intersects(gl.union_all().envelope)].plot(ax=ax, color='#f4f4f4', edgecolor='white', lw=0.2)
    for k, c in LCOL.items():
        q = gl[gl['a'] == k]
        if len(q):
            q.plot(ax=ax, color=c, edgecolor='white', lw=0.25)
    u = a3 == 'under-aligned'
    lvl_rows.append({'need_level': lab, **{k: int((a3 == k).sum()) for k in LCOL}, 'people_under_aligned': int(d.loc[u, 'population'].sum())})
    x0, y0, x1, y1 = gl.total_bounds
    ax.set_xlim(x0 - 0.05, x1 + 0.05); ax.set_ylim(y0 - 0.05, y1 + 0.05); ax.set_axis_off()
    ax.set_title(f"{lab}\nunder < {UNDER:g}x need, over > {OVER:g}x\nunder-aligned: {int(u.sum())} postcodes, "
                 f"{d.loc[u, 'population'].sum() / 1e6:.2f}M people", loc='left', fontsize=10)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in LCOL.items()], loc='lower center', ncol=4,
           frameon=False, fontsize=10)
fig.suptitle('Battery alignment with technical need at three need levels (all-segment battery kWh per solar site vs daily '
             'reverse flow per solar site)', x=0.01, ha='left', fontsize=12)
fig.tight_layout(rect=(0, 0.05, 1, 0.95)); fig.savefig(os.path.join(OUT, 'fig_7b_3_alignment_by_need_level.png'), dpi=170); plt.close(fig)
lv = pd.DataFrame(lvl_rows); lv.to_csv(os.path.join(OUT, '7b_alignment_by_need_level.csv'), index=False)
print('\nALIGNMENT BY NEED LEVEL'); print(lv.to_string(index=False))
print('Saved to', OUT)

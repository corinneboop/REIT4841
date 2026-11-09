### STEP 3.5 - Compare the six load-split methods with each other and with Energex's reported energy by postcode.

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MATCH_DIR = os.path.dirname(SCRIPT_DIR)
SPLIT = os.path.join(MATCH_DIR, '3_mesh_block_split', 'output')
VORONOI = os.path.join(MATCH_DIR, '4_voronoi_split', 'output')
METHODS = {
    'mb_equal': ('Mesh Block: equal per transformer', '#9ecae1', SPLIT),
    'mb_kva': ('Mesh Block: rated kVA', '#2a78d6', SPLIT),
    'mb_kva_utilisation': ('Mesh Block: kVA x utilisation', '#08306b', SPLIT),
    'voronoi_equal': ('Voronoi: equal per km2', '#fdae6b', VORONOI),
    'voronoi_kva': ('Voronoi: cells weighted by rated kVA', '#f16913', VORONOI),
    'voronoi_weighted': ('Voronoi: cells weighted by kVA x utilisation', '#d94801', VORONOI)}
SELECTED_METHOD = 'mb_kva_utilisation'
sys.path.insert(0, MATCH_DIR)
sys.dont_write_bytecode = True
from split_tools import MIN_MISSING_FRACTION, postcode_files
USAGE_FILE = os.path.join(MATCH_DIR, 'data', 'Energy-Usage-Postcode-31-Mar-2026.xlsx')
SITE_LOAD_FILE = os.path.join(os.path.dirname(MATCH_DIR), '1_join_substation_data', 'output', 'load_MW_by_site.csv')
COVERAGE_FILE = os.path.join(MATCH_DIR, '2_network_matching', 'output', 'network', 'postcode_coverage.csv')
MIN_COVERAGE = 0.9
HOURS_PER_INTERVAL = 0.5
WITHIN = 25
MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
               'November', 'December']
OUT = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT, exist_ok=True)
names = list(METHODS)

maps = {m: pd.read_csv(os.path.join(METHODS[m][2], m, 'site_postcode_mapping.csv'), dtype={'POSTCODE': str})
        for m in names}
site_all = pd.read_csv(SITE_LOAD_FILE, index_col='DATETIME', parse_dates=True)
cov = pd.read_csv(COVERAGE_FILE, dtype={'POSTCODE': str}).set_index('POSTCODE')

u = pd.read_excel(USAGE_FILE)
u = u[u['Measure Type'] == 'Total energy (kWh)'].melt(
    id_vars=['Post Code', 'Customer Consumption Type', 'Year'], value_vars=MONTH_NAMES, var_name='M',
    value_name='kWh')
u['MONTH'] = pd.to_datetime(u['Year'].astype(str) + '-' + u['M'], format='%Y-%B').dt.to_period('M')
u['POSTCODE'] = u['Post Code'].astype(int).astype(str).str.zfill(4)
rep = u.pivot_table(index=['POSTCODE', 'MONTH'], columns='Customer Consumption Type', values='kWh',
                    aggfunc='sum').fillna(0) / 1000
rep = rep.rename(columns={'Business': 'BUSINESS_MWh', 'Residential': 'RESIDENTIAL_MWh', 'Solar': 'SOLAR_EXPORT_MWh'})
rep['REPORTED_NET_MWh'] = rep['BUSINESS_MWh'] + rep['RESIDENTIAL_MWh'] - rep['SOLAR_EXPORT_MWh']
rep = rep[(rep['BUSINESS_MWh'] + rep['RESIDENTIAL_MWh']) > 0]

mon = site_all.index.to_period('M')
hours = pd.Series(site_all.index.days_in_month * 24, index=site_all.index).groupby(mon).first()
enough = site_all.notna().groupby(mon).mean() >= MIN_COVERAGE
def month_energy(x):
    return x.groupby(mon).mean().mul(hours, axis=0).where(enough)
metered = month_energy(site_all)
fwd_e = month_energy(site_all.clip(lower=0))
rev_e = month_energy(-site_all.clip(upper=0))
months = sorted(set(rep.index.get_level_values('MONTH')) & set(metered.index))
metered, fwd_e, rev_e = metered.loc[months], fwd_e.loc[months], rev_e.loc[months]


def matrix(mp, col='SHARE'):
    return mp.pivot_table(index='SITE', columns='POSTCODE', values=col, aggfunc='sum').fillna(0)


est_month, est_fwd, est_bus, est_oth = {}, {}, {}, {}
for m in names:
    S = matrix(maps[m])
    SB = matrix(maps[m].assign(V=maps[m]['SHARE'] * maps[m]['BUSINESS_FRACTION']), 'V').reindex_like(S)
    SO = matrix(maps[m].assign(V=maps[m]['SHARE'] * maps[m]['OTHER_FRACTION']), 'V').reindex_like(S)
    PF = matrix(maps[m], 'POSTCODE_FRACTION').reindex_like(S)
    SS = matrix(maps[m], 'SOLAR_SHARE').reindex_like(S)
    F, R = fwd_e.reindex(columns=S.index).fillna(0), rev_e.reindex(columns=S.index).fillna(0)
    ok = (metered.reindex(columns=S.index).isna().astype(float) @ PF) < MIN_MISSING_FRACTION
    est_month[m] = (F @ S - R @ SS).where(ok)
    est_fwd[m] = (F @ S).where(ok).sum(min_count=len(months))
    est_bus[m] = (F @ SB).where(ok).sum(min_count=len(months))
    est_oth[m] = (F @ SO).where(ok).sum(min_count=len(months))

tot = rep.loc[rep.index.get_level_values('MONTH').isin(months)].groupby('POSTCODE').sum()
tot['MONTHS_REPORTED'] = rep.loc[rep.index.get_level_values('MONTH').isin(months)].groupby('POSTCODE').size()
tot['REPORTED_BUSINESS_SHARE'] = tot['BUSINESS_MWh'] / (tot['BUSINESS_MWh'] + tot['RESIDENTIAL_MWh'])
for m in names:
    e = est_month[m].sum(min_count=len(months))
    tot[f'{m}_MWh'] = e
    tot[f'{m}_RATIO'] = e / tot['REPORTED_NET_MWh']
    tot[f'{m}_BUSINESS_SHARE'] = est_bus[m] / est_fwd[m]
    tot[f'{m}_BUS+OTHER_SHARE'] = (est_bus[m] + est_oth[m]) / est_fwd[m]
tot['ENERGEX_SHARE'] = cov['ENERGEX_SHARE'].reindex(tot.index)
outside = cov['OUTSIDE_ENERGEX'].reindex(tot.index)
tot['OUTSIDE_ENERGEX'] = outside.isna() | (outside == True)
tot = tot[tot[[f'{m}_MWh' for m in names]].notna().any(axis=1)]
keep = tot.index[~tot['OUTSIDE_ENERGEX'] & (tot['MONTHS_REPORTED'] == len(months)) & (tot['REPORTED_NET_MWh'] > 0)
                 & tot[[f'{m}_MWh' for m in names]].notna().all(axis=1)]
tot['SCORED'] = tot.index.isin(keep)
tot.round(4).to_csv(os.path.join(OUT, 'energy_check_by_postcode.csv'))
k = tot.loc[keep]
print(f"Months: {months[0]} - {months[-1]} ({len(months)}); postcodes with an estimate: {len(tot)}; scored: {len(keep)} "
      f"(left out: {tot['OUTSIDE_ENERGEX'].sum()} outside Energex, rest missing months or site data)")

rows = []
for m in names:
    r = k[f'{m}_RATIO']
    err = 100 * (r - 1).abs()
    bs, rb = k[f'{m}_BUSINESS_SHARE'], k['REPORTED_BUSINESS_SHARE']
    bo = k[f'{m}_BUS+OTHER_SHARE']
    lr = np.log(r)
    rows.append({
        'METHOD': m, 'POSTCODES': len(k),
        'OVERALL_RATIO': k[f'{m}_MWh'].sum() / k['REPORTED_NET_MWh'].sum(),
        'MEDIAN_RATIO': r.median(),
        'MEDIAN_ABS_ERROR_%': err.median(),
        f'PCT_WITHIN_{WITHIN}%': 100 * (err <= WITHIN).mean(),
        'PCT_OFF_BY_2x_OR_MORE': 100 * ((r >= 2) | (r <= 0.5)).mean(),
        'SPEARMAN_ENERGY': k[f'{m}_MWh'].corr(k['REPORTED_NET_MWh'], method='spearman'),
        'SPEARMAN_LOGRATIO_VS_REPORTED_BUSINESS_SHARE': lr.corr(rb, method='spearman'),
        'BUSINESS_SHARE_SPEARMAN': bs.corr(rb, method='spearman'),
        'BUSINESS_SHARE_PEARSON': bs.corr(rb),
        'BUSINESS_SHARE_MEDIAN_ABS_DIFF_PP': 100 * (bs - rb).abs().median(),
        'BUSINESS_SHARE_MEAN_DIFF_PP': 100 * (bs - rb).mean(),
        'BUS+OTHER_SHARE_SPEARMAN': bo.corr(rb, method='spearman'),
        'BUS+OTHER_SHARE_MEDIAN_ABS_DIFF_PP': 100 * (bo - rb).abs().median()})
site_e = metered.sum(min_count=len(months))
rep_net = tot['REPORTED_NET_MWh']
up_rows = {}
for m in names:
    mp = maps[m]
    ok_sites = mp.groupby('SITE')['POSTCODE'].apply(lambda x: x.isin(keep).all())
    ok_sites = ok_sites.index[ok_sites & site_e.reindex(ok_sites.index).notna()]
    agg = (mp['POSTCODE'].map(rep_net) * mp['POSTCODE_FRACTION']).groupby(mp['SITE']).sum()
    up_rows[m] = agg.reindex(ok_sites) / site_e.reindex(ok_sites)
    if m == 'voronoi_equal':
        pj = pd.read_csv(os.path.join(VORONOI, 'postcode_to_site.csv'), dtype={'POSTCODE': str})
        pj = pj[pj['POSTCODE'].isin(keep)]
        agg2 = pj.assign(E=pj['POSTCODE'].map(rep_net)).groupby('SITE')['E'].sum()
        up_rows[m + ' (many-to-one)'] = agg2.reindex(site_e.index).fillna(0).reindex(ok_sites) / site_e.reindex(ok_sites)
up = pd.DataFrame(up_rows)
up.round(4).to_csv(os.path.join(OUT, 'upward_check_by_site.csv'), index_label='SITE')
up_sum = pd.DataFrame({c: {'SITES': up[c].notna().sum(), 'MEDIAN_RATIO': up[c].median(),
                           'MEDIAN_ABS_ERROR_%': 100 * (up[c] - 1).abs().median(),
                           f'PCT_WITHIN_{WITHIN}%': 100 * ((up[c] - 1).abs() <= WITHIN / 100).mean(),
                           'SITES_GIVEN_NO_POSTCODE': int((up[c] == 0).sum())} for c in up}).T
up_sum.round(3).to_csv(os.path.join(OUT, 'upward_check_summary.csv'), index_label='METHOD')
print('\nC. Reported postcode energy summed up to zone substations vs metered:\n' + up_sum.round(2).to_string())
for r in rows:
    r['UPWARD_MEDIAN_ABS_ERROR_%'] = up_sum.loc[r['METHOD'], 'MEDIAN_ABS_ERROR_%']

summary = pd.DataFrame(rows)
summary.round(4).to_csv(os.path.join(OUT, 'method_comparison.csv'), index=False)
pd.set_option('display.width', 200)
print('\n' + summary.set_index('METHOD').round(3).T.to_string())

k_bins = pd.cut(k['REPORTED_BUSINESS_SHARE'], [0, .3, .5, .7, 1], include_lowest=True)
bins = k.groupby(k_bins, observed=True).agg(
    POSTCODES=('REPORTED_NET_MWh', 'size'),
    **{f'{m}_MEDIAN_RATIO': (f'{m}_RATIO', 'median') for m in names},
    **{f'{m}_MEDIAN_ABS_ERROR_%': (f'{m}_RATIO', lambda r: 100 * (r - 1).abs().median()) for m in names})
print('\nBy reported business share of consumption:\n' + bins.round(3).to_string())
diff = (k['mb_kva_utilisation_MWh'] / k['mb_kva_MWh'] - 1) * 100
print(f"\nmb_kva_utilisation vs mb_kva, change in postcode energy: median {diff.abs().median():.1f}%, "
      f"> 10% in {(diff.abs() > 10).sum()} postcodes, > 25% in {(diff.abs() > 25).sum()}")

def tidy(ax, axis='both'):
    ax.grid(axis=axis, color='#e0e0e0', linewidth=0.6); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


period = f"{months[0].strftime('%b %Y')} - {months[-1].strftime('%b %Y')}"
NC = 3; NR = -(-len(names) // NC)
fig, axes = plt.subplots(NR, NC, figsize=(5.6 * NC, 5.6 * NR), sharex=True, sharey=True)
for ax in axes.flat[len(names):]:
    ax.set_visible(False)
axes = axes.flat
lo, hi = max(1, k['REPORTED_NET_MWh'].min() / 1.5), k['REPORTED_NET_MWh'].max() * 1.5
for ax, (m, row) in zip(axes, summary.set_index('METHOD').iterrows()):
    lab, c, _ = METHODS[m]
    ax.scatter(k['REPORTED_NET_MWh'], k[f'{m}_MWh'].clip(lower=lo), s=14, color=c, alpha=0.75, edgecolor='white',
               linewidth=0.3)
    ax.plot([lo, hi], [lo, hi], color='#444444', lw=1, ls='--')
    for f in (0.5, 2):
        ax.plot([lo, hi], [lo * f, hi * f], color='#999999', lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(f"{lab}\nmedian error {row['MEDIAN_ABS_ERROR_%']:.0f}%, {row[f'PCT_WITHIN_{WITHIN}%']:.0f}% within "
                 f"{WITHIN}%, overall ratio {row['OVERALL_RATIO']:.2f}", loc='left', fontsize=10)
    ax.set_xlabel('Reported net energy (MWh)')
    tidy(ax)
for i in range(0, NR * NC, NC):
    axes[i].set_ylabel('Estimated from substation load (MWh, not rescaled)')
fig.suptitle(f'Postcode net energy, {period}, {len(k)} postcodes. Dashed = 1:1, dotted = 2x.', x=0.01, ha='left',
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(OUT, 'fig_1_energy_check.png'), dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.4))
for m in names:
    lab, c, _ = METHODS[m]
    ax.scatter(100 * k['REPORTED_BUSINESS_SHARE'], k[f'{m}_RATIO'].clip(0.25, 4), s=14, color=c, alpha=0.65, label=lab,
               edgecolor='white', linewidth=0.3)
ax.axhline(1, color='#444444', lw=1, ls='--')
ax.set_yscale('log'); ax.set_ylim(0.24, 4.2); ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(['0.25', '0.5', '1', '2', '4'])
ax.minorticks_off()
ax.set_xlabel('Reported business share of consumption (%)'); ax.set_ylabel('Estimate / reported net energy (clipped to 0.25 - 4)')
ax.set_title('Where the split goes wrong: ratio against how business-heavy the postcode is', loc='left', fontsize=10.5)
ax.legend(frameon=False)
tidy(ax)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_2_ratio_vs_business.png'), dpi=200); plt.close(fig)

NC = 3; NR = -(-len(names) // NC)
fig, axes = plt.subplots(NR, NC, figsize=(5.6 * NC, 5.6 * NR), sharex=True, sharey=True)
for ax in axes.flat[len(names):]:
    ax.set_visible(False)
axes = axes.flat
for ax, (m, row) in zip(axes, summary.set_index('METHOD').iterrows()):
    lab, c, _ = METHODS[m]
    ax.scatter(100 * k['REPORTED_BUSINESS_SHARE'], 100 * k[f'{m}_BUSINESS_SHARE'], s=14, color=c, alpha=0.75,
               edgecolor='white', linewidth=0.3)
    ax.plot([0, 100], [0, 100], color='#444444', lw=1, ls='--')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_title(f"{lab}\nSpearman {row['BUSINESS_SHARE_SPEARMAN']:.2f}, median |difference| "
                 f"{row['BUSINESS_SHARE_MEDIAN_ABS_DIFF_PP']:.0f} points", loc='left', fontsize=10)
    ax.set_xlabel('Reported business share of consumption (%)')
    tidy(ax)
for i in range(0, NR * NC, NC):
    axes[i].set_ylabel('Modelled business share (%)')
fig.suptitle(f'Business / residential split by postcode, {period}', x=0.01, ha='left', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(OUT, 'fig_3_business_share.png'), dpi=200)
plt.close(fig)

if SELECTED_METHOD:
    sel = os.path.join(OUT, 'selected')
    os.makedirs(sel, exist_ok=True)
    shutil.copy(os.path.join(METHODS[SELECTED_METHOD][2], SELECTED_METHOD, 'site_postcode_mapping.csv'),
                os.path.join(sel, 'site_postcode_mapping.csv'))
    load, reverse = postcode_files(maps[SELECTED_METHOD], site_all)
    load.round(4).to_csv(os.path.join(sel, 'load_MW_by_postcode.csv'))
    reverse.round(4).to_csv(os.path.join(sel, 'reverse_MW_by_postcode.csv'))

    in_months = reverse.index.to_period('M').isin(months)
    rv = reverse[in_months]
    complete = rv.notna().all()
    ex = pd.DataFrame({'REVERSE_MWh': rv.sum() * HOURS_PER_INTERVAL,
                       'REPORTED_SOLAR_EXPORT_MWh': tot['SOLAR_EXPORT_MWh'].reindex(rv.columns)})
    ex['PASS_THROUGH'] = ex['REVERSE_MWh'] / ex['REPORTED_SOLAR_EXPORT_MWh'].replace(0, np.nan)
    ex['CHECKED'] = complete & ex.index.isin(keep) & ex['PASS_THROUGH'].notna()
    ex['ABOVE_EXPORTS'] = ex['CHECKED'] & (ex['PASS_THROUGH'] > 1)
    ex.round(4).to_csv(os.path.join(OUT, 'export_check_by_postcode.csv'), index_label='POSTCODE')
    c = ex[ex['CHECKED']]
    print(f"\nD. Reverse flow / reported solar exports ({SELECTED_METHOD}, {len(c)} postcodes): network "
          f"{c['REVERSE_MWh'].sum() / c['REPORTED_SOLAR_EXPORT_MWh'].sum():.3f}, median {c['PASS_THROUGH'].median():.3f}, "
          f"P90 {c['PASS_THROUGH'].quantile(0.9):.3f}, max {c['PASS_THROUGH'].max():.2f} ({c['PASS_THROUGH'].idxmax()}); "
          f"above 1 (impossible): {', '.join(c.index[c['ABOVE_EXPORTS']]) or 'none'}")
    with open(os.path.join(sel, 'selected_method.txt'), 'w') as f:
        f.write(f'{SELECTED_METHOD}\n(set SELECTED_METHOD in 5_compare_methods.py to change)\n')
    print(f'\nHand-off written for {SELECTED_METHOD} to {sel}')
else:
    print('\nHand-off method not chosen yet (SELECTED_METHOD = None): nothing written for the later folders.')
print('Saved to', OUT)

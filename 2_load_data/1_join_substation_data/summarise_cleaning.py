### Step 1 results tables for the thesis (reads step 1 outputs; changes nothing).

import os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output'); TAB = os.path.join(OUT, 'cleaning_tables'); os.makedirs(TAB, exist_ok=True)
DT = 0.5
DAYLIGHT = list(range(6, 18))
NIGHT = [20, 21, 22, 23, 0, 1, 2, 3, 4]
EXCLUDE = {'Nth Pine Dam': 'Dam pumping station with no residential load: 0 MW for 57% of the year (median 0.00 MW, '
                           'maximum 1.3 MW)',
           'Amberley': 'Small flows recorded as zero: 37% of 09:00-14:30 readings exactly 0.00 MW vs ~1.4% per '
                       'adjacent 0.01 MW step, so reverse flow cannot be measured'}
ZERO_H, FLAT_H, NIGHT_H = 2, 4, 1
SENS = {'zero_run': [1, 1.5, 2, 3, 4, 6, 12, 24, 48], 'flat_run': [1, 1.5, 2, 3, 4, 6, 12, 24, 48],
        'night_reverse': [0.5, 1, 1.5, 2, 3, 4]}
LABEL = {'zero_run': 'Zero run', 'flat_run': 'Flat run', 'night_reverse': 'Night reverse'}

raw = pd.read_csv(os.path.join(OUT, 'load_MW_all_substations.csv'), index_col=0, parse_dates=True)
site_of = {c: re.sub(r'\s+Bus\s*\d+$', '', c).strip() for c in raw.columns}
allsite = raw.T.groupby(pd.Series(site_of)).sum(min_count=1).T.round(2)
site = allsite.drop(columns=list(EXCLUDE))
idx = site.index; hours = idx.hour.values; days = idx.normalize()
N_HH = site.notna().sum()
N_DAYS = site.notna().groupby(days).any().sum()
TOT_HH, TOT_DAYS = int(N_HH.sum()), int(N_DAYS.sum())
is_day = np.isin(hours, DAYLIGHT)


def find(kind, h):
    """Boolean mask (half-hours removed) and day mask (site-days removed) for one rule at threshold h."""
    m = np.zeros(site.shape, bool)
    nruns = pd.Series(0, index=site.columns)
    if kind == 'night_reverse':
        night = site[np.isin(hours, NIGHT)]
        rev_h = (night < 0).groupby(night.index.normalize()).sum() * DT
        flag = (rev_h >= h).reindex(days).fillna(False).values
        m = flag.copy()
        nruns = (rev_h >= h).sum()
    else:
        for j, s in enumerate(site.columns):
            v = site[s].values; ok = ~np.isnan(v)
            brk = np.r_[True, (v[1:] != v[:-1]) | ~ok[1:] | ~ok[:-1]]
            st = np.flatnonzero(brk); en = np.r_[st[1:], len(v)] - 1; n = en - st + 1
            sel = ok[st] & ((v[st] == 0) if kind == 'zero_run' else (v[st] != 0)) & (n * DT >= h)
            for a, b in zip(st[sel], en[sel]):
                m[a:b + 1, j] = True
            nruns[s] = int(sel.sum())
    M = pd.DataFrame(m, index=idx, columns=site.columns)
    D = (M & is_day[:, None]).groupby(days).any()
    return M, D, nruns


def energy(M):
    x = site.where(M)
    return x.clip(lower=0).sum() * DT, (-x.clip(upper=0)).sum() * DT


per = {}
masks = {}
for kind, h in [('zero_run', ZERO_H), ('flat_run', FLAT_H), ('night_reverse', NIGHT_H)]:
    M, D, nr = find(kind, h)
    masks[kind] = (M, D)
    fwd, rev = energy(M)
    t = pd.DataFrame({'RUNS' if kind != 'night_reverse' else 'DAYS_FLAGGED': nr,
                      'HOURS': M.sum() * DT, 'HALF_HOURS_REMOVED': M.sum(),
                      'PCT_OF_SITE_HALF_HOURS': 100 * M.sum() / N_HH,
                      'SITE_DAYS_REMOVED': D.sum(), 'PCT_OF_SITE_DAYS': 100 * D.sum() / N_DAYS,
                      'FORWARD_MWh_REMOVED': fwd, 'REVERSE_MWh_REMOVED': rev})
    if kind == 'night_reverse':
        night = site.where(np.isin(hours, NIGHT)[:, None] & M)
        t['NIGHT_REVERSE_HOURS'] = (night < 0).sum() * DT
        t['MAX_NIGHT_REVERSE_MW'] = (-night.clip(upper=0)).max()
    t = t[t.iloc[:, 0] > 0].sort_values(['SITE_DAYS_REMOVED', 'HALF_HOURS_REMOVED'], ascending=False)
    t.index.name = 'SUBSTATION'
    tot = t.sum(numeric_only=True); tot.name = 'TOTAL'
    tot['PCT_OF_SITE_HALF_HOURS'] = 100 * tot['HALF_HOURS_REMOVED'] / TOT_HH
    tot['PCT_OF_SITE_DAYS'] = 100 * tot['SITE_DAYS_REMOVED'] / TOT_DAYS
    if kind == 'night_reverse':
        tot['MAX_NIGHT_REVERSE_MW'] = t['MAX_NIGHT_REVERSE_MW'].max()
    t = pd.concat([t, tot.to_frame().T]).rename_axis('SUBSTATION')
    per[kind] = t

sens = {}
for kind, hs in SENS.items():
    rows = []
    for h in hs:
        M, D, nr = find(kind, h)
        fwd, rev = energy(M)
        pct = (100 * D.sum() / N_DAYS); pct = pct[pct > 0].sort_values(ascending=False).head(5)
        rows.append({'MIN_HOURS': h, 'RUNS_OR_DAYS': int(nr.sum()), 'SITES': int((nr > 0).sum()),
                     'HALF_HOURS_REMOVED': int(M.values.sum()), 'PCT_OF_ALL_HALF_HOURS': 100 * M.values.sum() / TOT_HH,
                     'SITE_DAYS_REMOVED': int(D.values.sum()), 'PCT_OF_ALL_SITE_DAYS': 100 * D.values.sum() / TOT_DAYS,
                     'FORWARD_MWh_REMOVED': fwd.sum(), 'REVERSE_MWh_REMOVED': rev.sum(),
                     'TOP_5_SITES_PCT_OF_SITE_DAYS': '; '.join(f'{s} ({v:.1f}%)' for s, v in pct.items())})
    sens[kind] = pd.DataFrame(rows)

runs = []
for s in site.columns:
    v = site[s].values; ok = ~np.isnan(v)
    brk = np.r_[True, (v[1:] != v[:-1]) | ~ok[1:] | ~ok[:-1]]
    st = np.flatnonzero(brk); en = np.r_[st[1:], len(v)] - 1
    k = ok[st]
    runs.append(pd.DataFrame({'SITE': s, 'V': v[st][k], 'N': (en - st + 1)[k]}))
R = pd.concat(runs, ignore_index=True)
zero, small, nz = R[R.V == 0], R[(R.V != 0) & (R.V.abs() <= 0.05)], R[R.V != 0]
step = site.diff()
p_rep = ((step == 0) & (site != 0) & site.shift().notna()).sum() / ((site != 0) & site.notna()).sum()
starts = nz.groupby('SITE').size()
t = sens['zero_run']
t['P_ZERO_RUN_LASTS'] = [100 * (zero.N * DT >= h).mean() for h in t.MIN_HOURS]
t['P_SMALL_READING_LASTS'] = [100 * (small.N * DT >= h).mean() for h in t.MIN_HOURS]
t['PCT_LIKELY_FAULTS'] = (100 * (1 - t.P_SMALL_READING_LASTS / t.P_ZERO_RUN_LASTS)).clip(lower=0)
t = sens['flat_run']
t['EXPECTED_BY_CHANCE'] = [float((starts * p_rep[starts.index] ** (h / DT - 1)).sum()) for h in t.MIN_HOURS]
t['PCT_LIKELY_FAULTS'] = (100 * (1 - t.EXPECTED_BY_CHANCE / t.RUNS_OR_DAYS)).clip(lower=0)
ev = pd.read_csv(os.path.join(HERE, 'sensitivity', 'output', 'night_reverse_events.csv'))
ev = ev[~ev.SITE.isin(list(EXCLUDE))]
t = sens['night_reverse']
t['PCT_ABOVE_BATTERY_CAPACITY'] = [100 * (ev[ev.NIGHT_REVERSE_H >= h].MAX_NIGHT_REVERSE_MW >
                                          ev[ev.NIGHT_REVERSE_H >= h].BATTERY_MW_BEHIND_SITE).mean() for h in t.MIN_HOURS]
t['PCT_ABOVE_NCC'] = [100 * (ev[ev.NIGHT_REVERSE_H >= h].PCT_OF_NCC > 100).mean() for h in t.MIN_HOURS]
for k in sens:
    c = 'TOP_5_SITES_PCT_OF_SITE_DAYS'; sens[k] = sens[k][[x for x in sens[k].columns if x != c] + [c]]

exc = []
for s, why in EXCLUDE.items():
    x = allsite[s]
    exc.append({'SUBSTATION': s, 'HALF_HOURS': int(x.notna().sum()), 'MEDIAN_MW': x.median(), 'MAX_MW': x.max(),
                'PCT_EXACTLY_ZERO': 100 * (x == 0).mean(),
                'PCT_ZERO_09_1430': 100 * (x[np.isin(x.index.hour, range(9, 15))] == 0).mean(),
                'FORWARD_MWh_YEAR': x.clip(lower=0).sum() * DT, 'REVERSE_MWh_YEAR': (-x.clip(upper=0)).sum() * DT,
                'REASON': why})
exc = pd.DataFrame(exc)

Mall = masks['zero_run'][0] | masks['flat_run'][0] | masks['night_reverse'][0]
Dall = masks['zero_run'][1] | masks['flat_run'][1] | masks['night_reverse'][1]
fwd, rev = energy(Mall)
tot_rows = []
for kind in ['zero_run', 'flat_run', 'night_reverse']:
    r = per[kind].loc['TOTAL']
    tot_rows.append({'RULE': f"{LABEL[kind]} (>= {dict(zero_run=ZERO_H, flat_run=FLAT_H, night_reverse=NIGHT_H)[kind]} h)",
                     'SUBSTATIONS': len(per[kind]) - 1, 'HALF_HOURS_REMOVED': int(r['HALF_HOURS_REMOVED']),
                     'PCT_OF_ALL_HALF_HOURS': r['PCT_OF_SITE_HALF_HOURS'], 'SITE_DAYS_REMOVED': int(r['SITE_DAYS_REMOVED']),
                     'PCT_OF_ALL_SITE_DAYS': r['PCT_OF_SITE_DAYS'], 'FORWARD_MWh_REMOVED': r['FORWARD_MWh_REMOVED'],
                     'REVERSE_MWh_REMOVED': r['REVERSE_MWh_REMOVED']})
tot_rows.append({'RULE': 'All three rules (overlaps counted once)', 'SUBSTATIONS': int(Mall.any().sum()),
                 'HALF_HOURS_REMOVED': int(Mall.values.sum()), 'PCT_OF_ALL_HALF_HOURS': 100 * Mall.values.sum() / TOT_HH,
                 'SITE_DAYS_REMOVED': int(Dall.values.sum()), 'PCT_OF_ALL_SITE_DAYS': 100 * Dall.values.sum() / TOT_DAYS,
                 'FORWARD_MWh_REMOVED': fwd.sum(), 'REVERSE_MWh_REMOVED': rev.sum()})
total = pd.DataFrame(tot_rows)
year_fwd = site.clip(lower=0).sum().sum() * DT; year_rev = (-site.clip(upper=0)).sum().sum() * DT
total['PCT_OF_FORWARD_ENERGY'] = 100 * total['FORWARD_MWh_REMOVED'] / year_fwd
total['PCT_OF_REVERSE_ENERGY'] = 100 * total['REVERSE_MWh_REMOVED'] / year_rev
info = pd.DataFrame([{'ITEM': 'Bus files', 'VALUE': raw.shape[1]}, {'ITEM': 'Zone substations', 'VALUE': allsite.shape[1]},
                     {'ITEM': 'Substations left out', 'VALUE': len(EXCLUDE)}, {'ITEM': 'Substations analysed', 'VALUE': site.shape[1]},
                     {'ITEM': 'Site half-hours analysed', 'VALUE': TOT_HH}, {'ITEM': 'Site-days analysed', 'VALUE': TOT_DAYS},
                     {'ITEM': 'Forward energy, analysed sites (MWh, raw)', 'VALUE': round(year_fwd)},
                     {'ITEM': 'Reverse energy, analysed sites (MWh, raw)', 'VALUE': round(year_rev)}])

sheets = {'summary_total': total, 'dataset': info, 'excluded_sites': exc,
          'zero_runs_by_site': per['zero_run'].reset_index(), 'flat_runs_by_site': per['flat_run'].reset_index(),
          'night_reverse_by_site': per['night_reverse'].reset_index(),
          'sens_zero_run': sens['zero_run'], 'sens_flat_run': sens['flat_run'], 'sens_night_reverse': sens['night_reverse']}
with pd.ExcelWriter(os.path.join(TAB, 'cleaning_tables.xlsx')) as xw:
    for name, df in sheets.items():
        df.round(3).to_csv(os.path.join(TAB, f'{name}.csv'), index=False)
        df.round(3).to_excel(xw, sheet_name=name, index=False)
pd.set_option('display.width', 400); pd.set_option('display.max_colwidth', 120); pd.set_option('display.max_rows', 200)
for name, df in sheets.items():
    print(f'\n### {name}'); print(df.round(2).to_string(index=False))

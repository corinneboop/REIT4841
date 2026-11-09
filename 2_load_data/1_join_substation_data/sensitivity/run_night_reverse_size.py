### Sensitivity of the night-reverse rule to SIZE instead of duration (does NOT change the pipeline).

import os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, LOAD); sys.dont_write_bytecode = True
import der_register
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
NIGHT = [20, 21, 22, 23, 0, 1, 2, 3, 4]
DAYLIGHT = list(range(6, 18))
EXCLUDE = ['Nth Pine Dam', 'Amberley']
DT = 0.5
ABS_MW = [0.5, 1, 2, 5, 10, 20]
PCT_NCC = [10, 25, 50, 100]
X_NIGHT = [0.5, 1, 2]

raw = pd.read_csv(os.path.join(LOAD, '1_join_substation_data', 'output', 'load_MW_all_substations.csv'),
                  index_col=0, parse_dates=True)
site_of = {c: re.sub(r'\s+Bus\s*\d+$', '', c).strip() for c in raw.columns}
site = raw.T.groupby(pd.Series(site_of)).sum(min_count=1).T.drop(columns=EXCLUDE)
hour, day = site.index.hour, site.index.normalize()

ncc = pd.read_csv(os.path.join(LOAD, '2_dapr_parameters', 'output', 'dapr_validation.csv')).groupby('SITE')['NCC_MVA_S2526'].sum(min_count=1)
deep = site[np.isin(hour, [0, 1, 2, 3])]
night_load = deep.clip(lower=0).median()
sp = pd.read_csv(os.path.join(LOAD, '3_postcode_matching', '2_network_matching', 'output', 'network', 'site_postcode_solar.csv'),
                 dtype={'POSTCODE': str})
bkw = der_register.der_by_postcode()['battery_kW']; bkw.index = bkw.index.astype(str)
batt_kw = (sp['POSTCODE'].map(bkw).fillna(0) * sp['DWELLING_FRACTION']).groupby(sp['SITE']).sum()

night = site[np.isin(hour, NIGHT)]
nd = night.index.normalize()
rev_h = (night < 0).groupby(nd).sum() * DT
rev_max = (-night.clip(upper=0)).groupby(nd).max()
day_rev = (-site.clip(upper=0)).groupby(day).sum() * DT
day_fwd = site.clip(lower=0).groupby(day).sum() * DT
rows = []
for s in site.columns:
    for d in rev_h.index[rev_h[s] > 0]:
        rows.append({'SITE': s, 'DATE': d.date(), 'NIGHT_REVERSE_H': rev_h.at[d, s], 'MAX_NIGHT_REVERSE_MW': rev_max.at[d, s],
                     'NCC_MVA': ncc.get(s, np.nan), 'TYPICAL_NIGHT_LOAD_MW': night_load[s],
                     'BATTERY_MW_BEHIND_SITE': batt_kw.get(s, 0) / 1000,
                     'DAY_REVERSE_MWh': day_rev.at[d, s], 'DAY_FORWARD_MWh': day_fwd.at[d, s]})
ev = pd.DataFrame(rows)
ev['PCT_OF_NCC'] = 100 * ev['MAX_NIGHT_REVERSE_MW'] / ev['NCC_MVA']
ev['X_NIGHT_LOAD'] = ev['MAX_NIGHT_REVERSE_MW'] / ev['TYPICAL_NIGHT_LOAD_MW'].replace(0, np.nan)
ev['X_BATTERY_MW'] = ev['MAX_NIGHT_REVERSE_MW'] / ev['BATTERY_MW_BEHIND_SITE'].replace(0, np.nan)
ev['CURRENT_RULE_1H'] = ev['NIGHT_REVERSE_H'] >= 1
ev.sort_values('MAX_NIGHT_REVERSE_MW', ascending=False).round(3).to_csv(os.path.join(OUT, 'night_reverse_events.csv'), index=False)
cur = set(zip(ev.loc[ev.CURRENT_RULE_1H, 'SITE'], ev.loc[ev.CURRENT_RULE_1H, 'DATE']))
print(f"Candidate site-days (any night reverse half-hour): {len(ev)} at {ev.SITE.nunique()} sites; "
      f"current 1 h rule flags {len(cur)}")

rules = [('duration (current)', '>= 1 h', ev['NIGHT_REVERSE_H'] >= 1)]
rules += [('absolute', f'>= {v:g} MW', ev['MAX_NIGHT_REVERSE_MW'] >= v) for v in ABS_MW]
rules += [('% of rating', f'>= {v}% NCC', ev['PCT_OF_NCC'] >= v) for v in PCT_NCC]
rules += [('x night load', f'>= {v:g} x', ev['X_NIGHT_LOAD'] >= v) for v in X_NIGHT]
out = []
for kind, lab, m in rules:
    f = ev[m.fillna(False)]
    keys = set(zip(f.SITE, f.DATE))
    out.append({'RULE': kind, 'THRESHOLD': lab, 'SITE_DAYS': len(f), 'SITES': f.SITE.nunique(),
                'OF_CURRENT_18_FLAGGED': len(keys & cur), 'NEW_vs_CURRENT': len(keys - cur),
                'REVERSE_MWh_REMOVED_WHOLE_DAY': f['DAY_REVERSE_MWh'].sum(), 'FORWARD_MWh_REMOVED': f['DAY_FORWARD_MWh'].sum(),
                'SITES_LIST': ', '.join(sorted(f.SITE.unique()))})
tab = pd.DataFrame(out)
tab.round(1).to_csv(os.path.join(OUT, 'sensitivity_night_reverse_size.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200)
print(tab.drop(columns='SITES_LIST').round(1).to_string(index=False))
for r in out:
    print(f"  {r['THRESHOLD']:>12s}: {r['SITES_LIST']}")
print('\nLargest 25 night events:')
print(ev.sort_values('MAX_NIGHT_REVERSE_MW', ascending=False).head(25)[['SITE', 'DATE', 'NIGHT_REVERSE_H', 'MAX_NIGHT_REVERSE_MW',
      'PCT_OF_NCC', 'X_NIGHT_LOAD', 'BATTERY_MW_BEHIND_SITE', 'CURRENT_RULE_1H']].round(2).to_string(index=False))
print('\nDistribution of MAX_NIGHT_REVERSE_MW over all candidates:', ev['MAX_NIGHT_REVERSE_MW'].describe(
      percentiles=[.5, .75, .9, .95]).round(2).to_dict())

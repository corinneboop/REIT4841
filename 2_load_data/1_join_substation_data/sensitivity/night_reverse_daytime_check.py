### Are the DAYTIME readings on night-reverse days also abnormal? (does NOT change the pipeline)
### Step 1 removes the whole site-day when reverse flow lasts >= 1 h at night (20:00-04:30). This checks the reason:
### for each such day, the daytime half-hours (06:00-17:30) are compared with the median profile of normal days at
### the same site within 14 days. DAY_ABNORMAL = median |deviation| > half the normal daytime range.

import os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', 'output', 'load_MW_all_substations.csv')
OUT = os.path.join(HERE, 'output'); os.makedirs(OUT, exist_ok=True)
DAYLIGHT = list(range(6, 18))
NIGHT = [20, 21, 22, 23, 0, 1, 2, 3, 4]
EXCLUDE = ['Nth Pine Dam', 'Amberley']
DT = 0.5

raw = pd.read_csv(RAW, index_col=0, parse_dates=True)
site_of = {c: re.sub(r'\s+Bus\s*\d+$', '', c).strip() for c in raw.columns}
site = raw.T.groupby(pd.Series(site_of)).sum(min_count=1).T.drop(columns=EXCLUDE)
hour = site.index.hour

night = site[np.isin(hour, NIGHT)]
n_rev = (night < 0).groupby(night.index.normalize()).sum() * DT

flag1 = n_rev >= 1
rows = []
for s in site.columns:
    for d in flag1.index[flag1[s]]:
        x = site[s]
        dd = x.loc[d:d + pd.Timedelta('23h30min')]
        ref_days = x[(x.index.normalize() != d) & (abs((x.index.normalize() - d).days) <= 14) & ~flag1[s].reindex(
            x.index.normalize()).fillna(False).values]
        day_h = dd[np.isin(dd.index.hour, DAYLIGHT)]
        ref_h = ref_days[np.isin(ref_days.index.hour, DAYLIGHT)]
        prof = ref_h.groupby(ref_h.index.time).median()
        dev = (day_h.values - prof.reindex(day_h.index.time).values)
        rows.append({'SITE': s, 'DATE': d.date(), 'NIGHT_REVERSE_H': n_rev.at[d, s], 'NIGHT_MIN_MW': dd[np.isin(dd.index.hour, NIGHT)].min(),
                     'DAY_MIN_MW': day_h.min(), 'NORMAL_DAY_MIN_MW': ref_h.groupby(ref_h.index.normalize()).min().median(),
                     'DAY_MEDIAN_ABS_DEV_MW': np.nanmedian(np.abs(dev)),
                     'NORMAL_DAY_RANGE_MW': prof.max() - prof.min()})
c = pd.DataFrame(rows)
c['DAY_ABNORMAL'] = c['DAY_MEDIAN_ABS_DEV_MW'] > 0.5 * c['NORMAL_DAY_RANGE_MW']
c.round(2).to_csv(os.path.join(OUT, 'night_reverse_daytime_check.csv'), index=False)
print('\nC. DAYTIME (06:00-17:30) ON NIGHT-REVERSE DAYS vs the median profile of normal days within 14 days')
print('   DAY_ABNORMAL = median |deviation| > half the normal daytime range')
print(c.round(2).to_string(index=False))

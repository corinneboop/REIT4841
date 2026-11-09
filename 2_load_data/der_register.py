### Shared loader for the AEMO DER register quarterly snapshots (2_load_data/data/DERR).
### Used by 4_reverse_flow, 6_battery_alignment and 3_social_data/2_analysis_dataset, so every
### step uses the same solar and battery numbers.

import os
import numpy as np
import pandas as pd

DERR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'DERR')
DERR_FILES = {'2025-07': 'July 2025 DERR data.xlsx', '2025-09': 'september-2025-derr-data.csv',
              '2025-12': 'December 2025 DERR Data.xlsx', '2026-03': 'March 2026 DERR data.xlsx'}
ALL_QUARTERS = list(DERR_FILES)
SOLAR_QUARTERS = ['2025-09', '2025-12', '2026-03']
FIELDS = {'Sum of Solar_Connections': 'solar_sites', 'Sum of Solar_capacity_kVA': 'solar_kVA',
          'Sum of Battery_Connections': 'battery_sites', 'Sum of Battery_capacity_kVA': 'battery_kW',
          'Sum of Battery_Storage_kVAh': 'battery_kWh'}
SOLAR_FIELDS = ['solar_sites', 'solar_kVA']
BATTERY_FIELDS = ['battery_sites', 'battery_kW', 'battery_kWh']
_cache = {}


def load_quarters(state='QLD'):
    """Long table: postcode, segment ('res' / 'other'), quarter and the FIELDS, summed within each file."""
    if state in _cache:
        return _cache[state]
    out = []
    for q, fn in DERR_FILES.items():
        path = os.path.join(DERR_DIR, fn)
        t = (pd.read_excel(path) if fn.endswith('.xlsx') else pd.read_csv(path, encoding='utf-8-sig'))
        t.columns = ['state', 'postcode', 'segment'] + list(t.columns[3:])
        t = t[t['state'] == state].copy()
        t['postcode'] = t['postcode'].astype(str).str.replace(r'\.0$', '', regex=True)
        t['segment'] = np.where(t['segment'].astype(str).str.upper() == 'RESIDENTIAL', 'res', 'other')
        g = t.groupby(['postcode', 'segment'])[list(FIELDS)].sum().rename(columns=FIELDS)
        g['quarter'] = q
        out.append(g.reset_index())
    _cache[state] = pd.concat(out, ignore_index=True)
    return _cache[state]


def mean_over(fields, quarters, segment=None, state='QLD'):
    """Per postcode, the mean over the given quarters (all segments, or one segment)."""
    x = load_quarters(state)
    x = x[x['quarter'].isin(quarters)]
    if segment:
        x = x[x['segment'] == segment]
    per_q = x.groupby(['postcode', 'quarter'])[fields].sum().unstack('quarter').fillna(0)
    return per_q.T.groupby(level=0).mean().T


def der_by_postcode(segment=None, state='QLD'):
    """Solar (Sep-Mar mean) and batteries (four-quarter mean) per postcode, for all segments or one segment."""
    solar = mean_over(SOLAR_FIELDS, SOLAR_QUARTERS, segment, state)
    batt = mean_over(BATTERY_FIELDS, ALL_QUARTERS, segment, state)
    return solar.join(batt, how='outer').fillna(0)


def quarter_table(fields, segment=None, state='QLD'):
    """Per postcode, one column per quarter (for growth between snapshots)."""
    x = load_quarters(state)
    if segment:
        x = x[x['segment'] == segment]
    return x.pivot_table(index='postcode', columns='quarter', values=fields, aggfunc='sum')

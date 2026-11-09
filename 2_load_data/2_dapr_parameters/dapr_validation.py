### Validate the Energex load data against the Energex DAPR 2025 zone substation workbook.

import os
import re
import numpy as np
import pandas as pd
import openpyxl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FOLDER = os.path.join(SCRIPT_DIR, '..', '1_join_substation_data', 'raw_load_data')
DAPR_FILE = os.path.join(SCRIPT_DIR, 'data', 'Zone_Substations_Load_Forecast_-_Energex_2025.xlsx')
QUALITY_FILE = os.path.join(SCRIPT_DIR, '..', '1_join_substation_data', 'output', 'load_quality_flags.csv')
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'output')

SUMMER_MONTHS = [11, 12, 1, 2, 3]
WINTER_MONTHS = [7, 8]
WINTER_YEAR = 2025
RATIO_OK = (0.8, 1.2)
RATIO_MODERATE = (0.67, 1.5)
PF_TOLERANCE = 0.05
SMALL_MVA = 2.0
SPIKE_FACTOR = 2.0
HOURS_PER_INTERVAL = 0.5
DAPR_SEASONS = ['S2526', 'S2627', 'S2728', 'S2829', 'S2930', 'W2025', 'W2026', 'W2027', 'W2028', 'W2029']

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def norm(name):
    """Lower case, letters and digits only, for matching names."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def site_name(name):
    """Remove the bus number: 'Bald Hills Bus 2' -> 'Bald Hills' (same rule as 3_postcode_matching)."""
    return re.sub(r'\s+Bus\s*\d+$', '', str(name)).strip()


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


print('Reading the DAPR workbook...')
wb = openpyxl.load_workbook(DAPR_FILE, read_only=True, data_only=True)
dapr_rows = []
for sheet in wb.sheetnames[2:]:
    rec = {'CODE': sheet}
    season_cols = None
    for r in wb[sheet].iter_rows(values_only=True):
        cells = [(j, v) for j, v in enumerate(r) if v is not None and str(v).strip() != '']
        if not cells:
            continue
        label = str(cells[0][1]).strip()
        rest = [v for _, v in cells[1:]]
        if label == 'Substation':
            full = str(rest[0]) if rest else ''
            rec['DAPR_FULL_NAME'] = full
            rec['DAPR_NAME'] = full.split(' - ', 1)[-1].rsplit(' (', 1)[0]
            rec['SUPPLY'] = rest[2] if len(rest) > 2 else None
        elif label.startswith('Transformer size'):
            rec['TRANSFORMERS'] = label
        elif label.startswith('Capacity of commissioned Embedded Generation'):
            m = re.search(r'is\s*([\d.]+)\s*MVA', label)
            rec['COMMISSIONED_EG_MVA'] = float(m.group(1)) if m else np.nan
        elif label == 'Growth Rates':
            rec['GROWTH_RATES'] = rest[0] if rest else None
        elif label == 'Load Category':
            rec['LOAD_CATEGORY'] = rest[0] if rest else None
        elif label == 'Latest Peak Compensated Load':
            txt = str(rest[0]) if rest else ''
            rec['DAPR_LATEST_PEAK_TEXT'] = txt
            m = re.search(r'([\d.]+)\s*MVA on (\d+/\d+/\d+) at (\d+:\d+)', txt)
            if m:
                rec['DAPR_LATEST_PEAK_MVA'] = float(m.group(1))
                rec['DAPR_LATEST_PEAK_TIME'] = pd.to_datetime(m.group(2) + ' ' + m.group(3),
                                                              format='%d/%m/%Y %H:%M')
        elif label == 'Major Loads':
            rec['MAJOR_LOADS'] = rest[0] if rest else None
        elif label == 'PEAK LOAD FORECAST AND CAPACITY':
            season_cols = 'next'
        elif season_cols == 'next':
            season_cols = [j for j, _ in cells]
        elif isinstance(season_cols, list) and len(season_cols) == 10 and cells[0][0] < season_cols[0]:
            key = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').upper()
            for col, season in zip(season_cols, DAPR_SEASONS):
                v = r[col] if col < len(r) else None
                rec[f'{key}__{season}'] = v
    dapr_rows.append(rec)

dapr = pd.DataFrame(dapr_rows)
dapr.to_csv(os.path.join(OUTPUT_FOLDER, 'dapr_parameters_by_bus.csv'), index=False)
print(f'  {len(dapr)} substation sheets parsed')

F = {
    'NCC': 'NCC_RATING_MVA',
    'POE10': '10_POE_LOAD_MVA',
    'POE50': '50_POE_LOAD_MVA',
    'PF': 'POWER_FACTOR_AT_PEAK_LOAD',
    'EG': 'CAPACITY_FOR_EMBEDDED_GENERATION_MVA',
    'EGN1': 'N_1_CAPACITY_FOR_EMBEDDED_GENERATION_MVA',
}
for k, v in F.items():
    for s in ['S2526', 'W2025']:
        col = f'{v}__{s}'
        if col not in dapr.columns:
            raise SystemExit(f'Field not found in workbook: {col}')

files = {f.split('_EGX_')[0]: f for f in os.listdir(RAW_FOLDER) if f.endswith('.csv')}
file_by_norm = {norm(n): n for n in files}
dapr['ENERGEX_NAME'] = dapr['DAPR_NAME'].map(lambda n: file_by_norm.get(norm(n), file_by_norm.get(norm(f'{n} Bus 1'))))
dapr['SITE'] = dapr['ENERGEX_NAME'].map(lambda n: site_name(n) if isinstance(n, str) else None)


if not os.path.exists(QUALITY_FILE):
    raise SystemExit('Run 1_join_substation_data/join_load_data.py first (needed for the quality runs).')
runs = pd.read_csv(QUALITY_FILE, parse_dates=['RUN_START', 'RUN_END'])


def in_runs(index, sub):
    mask = pd.Series(False, index=index)
    for _, run in runs[runs['SITE'] == site_name(sub)].iterrows():
        mask |= (index >= run['RUN_START']) & (index <= run['RUN_END'])
    return mask.values


def band(ratio):
    if pd.isna(ratio):
        return 'n/a'
    if RATIO_OK[0] <= ratio <= RATIO_OK[1]:
        return 'ok'
    if RATIO_MODERATE[0] <= ratio <= RATIO_MODERATE[1]:
        return 'moderate'
    return 'severe'


def peak(df):
    """Largest MVA among forward-flow intervals; returns (MVA, MW, time)."""
    fwd = df[(df['MW'] > 0) & df['MVA'].notna()]
    if fwd.empty:
        return np.nan, np.nan, pd.NaT
    t = fwd['MVA'].idxmax()
    return fwd.at[t, 'MVA'], fwd.at[t, 'MW'], t


def spike(daily):
    """Ratio of the largest daily value to the second largest (positive values only)."""
    top = daily[daily > 0].nlargest(2)
    return top.iloc[0] / top.iloc[1] if len(top) == 2 and top.iloc[1] > 0 else np.nan


print('Reading the Energex load files...')
out = []
bus_load = {}
for _, d in dapr[dapr['ENERGEX_NAME'].notna()].iterrows():
    name = d['ENERGEX_NAME']
    raw = pd.read_csv(os.path.join(RAW_FOLDER, files[name]), encoding='utf-8-sig', dtype=str)
    time = pd.to_datetime(raw['Date'] + ' ' + raw['Time'], format='%Y-%m-%d %H:%M:%S')
    df = pd.DataFrame({'MW': pd.to_numeric(raw['MW'], errors='coerce').values,
                       'MVA': pd.to_numeric(raw['MVA'], errors='coerce').values}, index=time).sort_index()
    df = df[~df.index.duplicated()]
    bad = in_runs(df.index, name)
    clean = df[~bad]
    bus_load[name] = clean

    summer = clean[clean.index.month.isin(SUMMER_MONTHS)]
    winter = clean[clean.index.month.isin(WINTER_MONTHS) & (clean.index.year == WINTER_YEAR)]

    a_mva, a_mw, a_t = peak(clean)
    s_mva, s_mw, s_t = peak(summer)
    w_mva, w_mw, w_t = peak(winter)
    a_mva_all, _, a_t_all = peak(df)

    p50_s, p50_w = to_float(d[f"{F['POE50']}__S2526"]), to_float(d[f"{F['POE50']}__W2025"])
    p10_s, p10_w = to_float(d[f"{F['POE10']}__S2526"]), to_float(d[f"{F['POE10']}__W2025"])
    pf_s, pf_w = to_float(d[f"{F['PF']}__S2526"]), to_float(d[f"{F['PF']}__W2025"])
    gov = 'Summer' if (np.nan_to_num(p50_s) >= np.nan_to_num(p50_w)) else 'Winter'
    p50_gov = p50_s if gov == 'Summer' else p50_w
    p10_gov = p10_s if gov == 'Summer' else p10_w
    obs_gov, obs_gov_t, obs_gov_mw = (s_mva, s_t, s_mw) if gov == 'Summer' else (w_mva, w_t, w_mw)
    pf_gov = pf_s if gov == 'Summer' else pf_w

    obs_pf = obs_gov_mw / obs_gov if obs_gov else np.nan

    rev = -clean['MW'].clip(upper=0)
    rev_t = rev.idxmax() if rev.max() > 0 else pd.NaT
    daily_rev = rev.groupby(rev.index.date).max()

    out.append({
        'CODE': d['CODE'], 'DAPR_NAME': d['DAPR_NAME'], 'ENERGEX_NAME': name, 'SITE': d['SITE'],
        'LOAD_CATEGORY': d['LOAD_CATEGORY'], 'COMMISSIONED_EG_MVA': d.get('COMMISSIONED_EG_MVA'),
        'NCC_MVA_S2526': to_float(d[f"{F['NCC']}__S2526"]),
        'DAPR_LATEST_PEAK_MVA': d.get('DAPR_LATEST_PEAK_MVA'), 'DAPR_LATEST_PEAK_TIME': d.get('DAPR_LATEST_PEAK_TIME'),
        'OBS_PEAK_MVA': a_mva, 'OBS_PEAK_MW': a_mw, 'OBS_PEAK_TIME': a_t,
        'RATIO_OBS_TO_LATEST': a_mva / d['DAPR_LATEST_PEAK_MVA'] if d.get('DAPR_LATEST_PEAK_MVA') else np.nan,
        'DAPR_50POE_S2526': p50_s, 'DAPR_10POE_S2526': p10_s, 'OBS_SUMMER_PEAK_MVA': s_mva, 'OBS_SUMMER_PEAK_TIME': s_t,
        'RATIO_SUMMER_TO_50POE': s_mva / p50_s if p50_s else np.nan,
        'DAPR_50POE_W2025': p50_w, 'DAPR_10POE_W2025': p10_w, 'OBS_WINTER_PEAK_MVA': w_mva, 'OBS_WINTER_PEAK_TIME': w_t,
        'RATIO_WINTER_TO_50POE': w_mva / p50_w if p50_w else np.nan,
        'GOVERNING_SEASON': gov, 'RATIO_GOV_TO_50POE': obs_gov / p50_gov if p50_gov else np.nan,
        'ABOVE_10POE': bool(obs_gov > p10_gov) if p10_gov and not np.isnan(obs_gov) else False,
        'DAPR_PF': pf_gov, 'OBS_PF_AT_PEAK': obs_pf,
        'DAPR_EG_CAPACITY_MVA': to_float(d[f"{F['EG']}__S2526"]),
        'DAPR_EG_N1_CAPACITY_MVA': to_float(d[f"{F['EGN1']}__S2526"]),
        'OBS_MAX_REVERSE_MW': rev.max(), 'OBS_MAX_REVERSE_TIME': rev_t,
        'REVERSE_HOURS': (rev > 0).sum() * HOURS_PER_INTERVAL,
        'REVERSE_SPIKE_RATIO': spike(daily_rev),
        'PEAK_IN_RUN': bool(pd.notna(a_t_all) and in_runs(pd.DatetimeIndex([a_t_all]), name)[0]),
        'OBS_PEAK_MVA_INCL_RUNS': a_mva_all,
        'INTERVALS_EXCLUDED': int(bad.sum()),
        'REVERSE_FLOW_INTERVALS': int((df['MW'] < 0).sum()),
    })

v = pd.DataFrame(out)
v['PEAK_FLAG'] = v['RATIO_GOV_TO_50POE'].map(band)
v['LATEST_FLAG'] = v['RATIO_OBS_TO_LATEST'].map(band)
v['PF_FLAG'] = (v['OBS_PF_AT_PEAK'] - v['DAPR_PF']).abs() > PF_TOLERANCE
v['SMALL_SUB'] = v[['DAPR_50POE_S2526', 'DAPR_50POE_W2025']].max(axis=1) < SMALL_MVA
v['RATIO_REVERSE_TO_EG'] = v['OBS_MAX_REVERSE_MW'] / v['DAPR_EG_CAPACITY_MVA'].replace(0, np.nan)
v['REVERSE_ABOVE_NCC'] = v['OBS_MAX_REVERSE_MW'] > v['NCC_MVA_S2526']
v['REVERSE_ABOVE_EG'] = v['OBS_MAX_REVERSE_MW'] > v['DAPR_EG_CAPACITY_MVA']
v['REVERSE_SPIKE'] = (v['REVERSE_SPIKE_RATIO'] > SPIKE_FACTOR) & (v['OBS_MAX_REVERSE_MW'] > 1)
v['NO_DAPR_FORECAST'] = v['DAPR_50POE_S2526'].isna() & v['DAPR_50POE_W2025'].isna()

severity = {'severe': 3, 'moderate': 2, 'ok': 0, 'n/a': 0}
v['FLAG_SCORE'] = (v['PEAK_FLAG'].map(severity) * 10 + v['LATEST_FLAG'].map(severity) * 2
                   + v['ABOVE_10POE'] * 3 + v['PF_FLAG'] + v['PEAK_IN_RUN'] * 2
                   + v['REVERSE_ABOVE_NCC'] * 25 + v['REVERSE_SPIKE'] * 8)


def reasons(r):
    out = []
    if r['NO_DAPR_FORECAST']:
        out.append('no DAPR forecast')
    if r['PEAK_FLAG'] in ('moderate', 'severe'):
        out.append(f"{r['GOVERNING_SEASON'].lower()} peak {r['RATIO_GOV_TO_50POE']:.2f}x 50 PoE ({r['PEAK_FLAG']})")
    if r['ABOVE_10POE']:
        out.append('above 10 PoE')
    if r['LATEST_FLAG'] in ('moderate', 'severe'):
        out.append(f"{r['RATIO_OBS_TO_LATEST']:.2f}x latest peak ({r['LATEST_FLAG']})")
    if r['PF_FLAG']:
        out.append(f"PF {r['OBS_PF_AT_PEAK']:.2f} vs {r['DAPR_PF']:.2f}")
    if r['PEAK_IN_RUN']:
        out.append('peak inside zero/flat run')
    if r['REVERSE_ABOVE_NCC']:
        out.append(f"reverse flow {r['OBS_MAX_REVERSE_MW']:.1f} MW > NCC {r['NCC_MVA_S2526']:.1f} MVA (bad data?)")
    if r['REVERSE_SPIKE']:
        out.append(f"one-day reverse spike {r['OBS_MAX_REVERSE_MW']:.1f} MW ({r['REVERSE_SPIKE_RATIO']:.1f}x next day) "
                   f"on {pd.Timestamp(r['OBS_MAX_REVERSE_TIME']):%Y-%m-%d %H:%M}")
    if out and r['SMALL_SUB']:
        out.append(f'small substation (< {SMALL_MVA:g} MVA)')
    return '; '.join(out)


v['FLAG_REASONS'] = v.apply(reasons, axis=1)
v = v.sort_values(['FLAG_SCORE', 'CODE'], ascending=[False, True])
v.to_csv(os.path.join(OUTPUT_FOLDER, 'dapr_validation.csv'), index=False)
print(f"dapr_validation.csv: {len(v)} matched buses, {int((v['FLAG_REASONS'] != '').sum())} with a flag; reverse flow above NCC on {int(v['REVERSE_ABOVE_NCC'].sum())} buses")
print(f'Outputs saved to {OUTPUT_FOLDER}')

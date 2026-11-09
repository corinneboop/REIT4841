### STEP 1b - Join the QLDSOUTH rows from every half-hourly AEMO rooftop PV CSV, and build the final dataset.

import os
import csv
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, 'raw_pv_data')
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'output')
AREA = 'QLDSOUTH'
INTERVAL_HOURS = 0.5
START_DATE, END_DATE = '2025-07-01', '2026-06-30'

TIMELINE = pd.date_range(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE) + pd.Timedelta('23h30min'),
                         freq='30min', name='INTERVAL_START')

SOURCES = {
    'PUBLIC_ROOFTOP_PV_ACTUAL_MEASUREMENT_AREA_': ('MEASURED', 'QLDSOUTH_measured_actuals.csv'),
    'PUBLIC_ROOFTOP_PV_ACTUAL_SATELLITE_AREA_': ('SATELLITE', 'QLDSOUTH_satellite_actuals.csv'),
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
tables = {}
gap_rows = []

for prefix, (source, output_name) in SOURCES.items():

    rows = []
    files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.startswith(prefix) and f.endswith('.csv'))

    for filename in files:
        with open(os.path.join(INPUT_FOLDER, filename), newline='') as f:
            for line in csv.reader(f):
                if line and line[0] == 'D' and line[5] == AREA:
                    rows.append({
                        'INTERVAL_END': line[4],
                        'POWER_MW': line[6],
                        'QI': line[7],
                    })

    df = pd.DataFrame(rows)
    df['INTERVAL_END'] = pd.to_datetime(df['INTERVAL_END'], format='%Y/%m/%d %H:%M:%S')
    df['POWER_MW'] = pd.to_numeric(df['POWER_MW'])
    df['QI'] = pd.to_numeric(df['QI'])
    df = df.drop_duplicates('INTERVAL_END', keep='last')

    df['INTERVAL_START'] = df['INTERVAL_END'] - pd.Timedelta('30min')
    df = df.drop(columns='INTERVAL_END').set_index('INTERVAL_START').sort_index()

    df['STATUS'] = 'ok'
    df.loc[df['POWER_MW'].isna(), 'STATUS'] = 'blank'

    n_raw = len(df)
    df = df.reindex(TIMELINE)
    df['STATUS'] = df['STATUS'].fillna('absent')

    df['ENERGY_MWH'] = df['POWER_MW'] * INTERVAL_HOURS
    df = df[['POWER_MW', 'ENERGY_MWH', 'QI', 'STATUS']]

    missing = df['STATUS'] != 'ok'
    run_id = (df['STATUS'] != df['STATUS'].shift()).cumsum()
    for _, run in df[missing].groupby(run_id[missing]):
        gap_rows.append({
            'SOURCE': source,
            'TYPE': run['STATUS'].iloc[0],
            'FIRST_INTERVAL_START': run.index.min(),
            'LAST_INTERVAL_END': run.index.max() + pd.Timedelta('30min'),
            'N_INTERVALS': len(run),
            'DAYS_REMOVED': ' '.join(sorted({d.strftime('%Y-%m-%d') for d in run.index.normalize()})),
        })

    df.to_csv(os.path.join(OUTPUT_FOLDER, output_name))
    tables[source] = df
    print(f"\n{output_name}: {len(files)} files read ({n_raw} intervals), {len(df)} half-hour intervals kept in "
          f"study period ({df.index.min()} to {df.index.max()}, interval start)")
    print(df['STATUS'].value_counts().to_string())

joined = pd.concat({s: tables[s] for s in ['MEASURED', 'SATELLITE']}, axis=1)
joined.columns = [f'{source}_{col}' for source, col in joined.columns]
joined.index.name = 'INTERVAL_START'
joined.to_csv(os.path.join(OUTPUT_FOLDER, 'QLDSOUTH_pv_joined.csv'))
print(f"\nQLDSOUTH_pv_joined.csv: {len(joined)} intervals, columns = {list(joined.columns)}")

gaps = pd.DataFrame(gap_rows, columns=['SOURCE', 'TYPE', 'FIRST_INTERVAL_START', 'LAST_INTERVAL_END',
                                       'N_INTERVALS', 'DAYS_REMOVED'])
gaps.to_csv(os.path.join(OUTPUT_FOLDER, 'missing_intervals.csv'), index=False)
print('\nMissing data (AEST; first interval START to last interval END):')
print(gaps.to_string(index=False) if len(gaps) else 'none')

reasons = []
for d, g in joined.groupby(joined.index.normalize()):
    why = []
    for src in ['MEASURED', 'SATELLITE']:
        counts = g.loc[g[f'{src}_STATUS'] != 'ok', f'{src}_STATUS'].value_counts()
        why += [f'{src.lower()} {status} x{n}' for status, n in counts.items()]
    if why:
        reasons.append({'DATE': d.date(), 'REASON': '; '.join(why)})
removed = pd.DataFrame(reasons, columns=['DATE', 'REASON'])
removed.to_csv(os.path.join(OUTPUT_FOLDER, 'removed_days.csv'), index=False)
print('\nDays removed from BOTH final datasets:')
print(removed.to_string(index=False) if len(removed) else 'none')

bad_days = pd.to_datetime(removed['DATE'])
for source, output_name in [('MEASURED', 'QLDSOUTH_measured_final.csv'),
                            ('SATELLITE', 'QLDSOUTH_satellite_final.csv')]:
    df = tables[source]
    keep = ~df.index.normalize().isin(bad_days)
    df.loc[keep].drop(columns='STATUS').to_csv(os.path.join(OUTPUT_FOLDER, output_name))
    print(f"{output_name}: {int(keep.sum())} intervals kept ({int(keep.sum()) // 48} days), "
          f"{int((~keep).sum())} dropped ({len(removed)} days)")

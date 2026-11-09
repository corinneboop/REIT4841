### STEP 3.1 - Queensland Mesh Block table: dwellings, land use, area, suburb (SAL) and postcode (POA).

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, '..', '..', 'data')
COUNTS_FILE = os.path.join(DATA, 'Mesh Block Counts, 2021.xlsx')
SAL_FILE = os.path.join(DATA, 'SAL_2021_AUST.xlsx')
POA_FILE = os.path.join(DATA, 'POA_2021_AUST.xlsx')
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'output')
CACHE = os.path.join(OUTPUT_FOLDER, 'cache')
QLD_TABLES = ['Table 3', 'Table 3.1']
os.makedirs(CACHE, exist_ok=True)


def cached(name, build):
    """Read output/cache/<name>.csv, or build it with build() and save it."""
    path = os.path.join(CACHE, f'{name}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={'MB_CODE': str, 'SAL_CODE': str, 'POSTCODE': str})
        if len(df):
            return df
    df = build()
    df.to_csv(path, index=False)
    print(f'  cached {name}: {len(df):,} rows')
    return df


def read_counts():
    parts = []
    for sheet in QLD_TABLES:
        d = pd.read_excel(COUNTS_FILE, sheet_name=sheet, header=6, dtype={'MB_CODE_2021': str})
        d = d[d['MB_CODE_2021'].str.fullmatch(r'\d{11}', na=False)]
        parts.append(d)
    d = pd.concat(parts)
    return pd.DataFrame({'MB_CODE': d['MB_CODE_2021'], 'MB_CATEGORY': d['MB_CATEGORY_NAME_2021'],
                         'AREA_SQKM': pd.to_numeric(d['AREA_ALBERS_SQKM']),
                         'DWELLINGS': pd.to_numeric(d['Dwelling'])})


def read_sal():
    d = pd.read_excel(SAL_FILE, dtype={'MB_CODE_2021': str, 'SAL_CODE_2021': str},
                      usecols=['MB_CODE_2021', 'SAL_CODE_2021', 'SAL_NAME_2021'])
    d = d[d['MB_CODE_2021'].str.startswith('3')]
    return d.rename(columns={'MB_CODE_2021': 'MB_CODE', 'SAL_CODE_2021': 'SAL_CODE',
                             'SAL_NAME_2021': 'SAL_NAME'})[['MB_CODE', 'SAL_CODE', 'SAL_NAME']]


def read_poa():
    d = pd.read_excel(POA_FILE, dtype={'MB_CODE_2021': str, 'POA_CODE_2021': str},
                      usecols=['MB_CODE_2021', 'POA_CODE_2021'])
    d = d[d['MB_CODE_2021'].str.startswith('3')]
    return pd.DataFrame({'MB_CODE': d['MB_CODE_2021'], 'POSTCODE': d['POA_CODE_2021'].str.zfill(4)})


print('Reading ABS files (cached after the first run)...')
counts = cached('mb_counts_QLD', read_counts)
sal = cached('mb_sal_QLD', read_sal)
poa = cached('mb_poa_QLD', read_poa)

mb = counts.merge(sal, on='MB_CODE', how='left').merge(poa, on='MB_CODE', how='left')
print(f"{len(mb):,} QLD Mesh Blocks; without SAL: {mb['SAL_CODE'].isna().sum()}, "
      f"without POA: {mb['POSTCODE'].isna().sum()} (these are usually 'No usual address'/offshore/migratory)")
mb[['MB_CODE', 'MB_CATEGORY', 'AREA_SQKM', 'DWELLINGS', 'SAL_NAME', 'POSTCODE']].to_csv(
    os.path.join(OUTPUT_FOLDER, 'mesh_blocks_QLD.csv'), index=False)

mb = mb.dropna(subset=['SAL_NAME', 'POSTCODE'])
print('Mesh Block categories (QLD, with a suburb and postcode):')
print(mb.groupby('MB_CATEGORY').agg(mesh_blocks=('MB_CODE', 'size'), dwellings=('DWELLINGS', 'sum'),
                                    area_sqkm=('AREA_SQKM', 'sum')).round(0).to_string())

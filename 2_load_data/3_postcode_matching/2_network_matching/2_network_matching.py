### STEP 3.2 - Network matching: which zone substation supplies each distribution transformer and Mesh Block,
### from the Energex network map. This step only builds the links; how a site's load is split between the
### transformers (and so between postcodes) is step 3.3 (3_mesh_block_split).

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings('ignore', category=UserWarning)

CRS_M = 7856
FEEDER_MAX_M = 1500
TRANSFORMER_MAX_M = 1000
MESHBLOCK_MIN_M, MESHBLOCK_MAX_M = 600, 5000
SEQ_BOX = (151.8, -28.45, 153.65, -25.75)
OUTSIDE_MAX = 0.5
MANUAL_SITE = {'GHM': 'Grantham',
               'T75': 'Nerang'}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MATCH_DIR = os.path.dirname(SCRIPT_DIR)
LOAD_DIR = os.path.dirname(MATCH_DIR)
GIS = os.path.join(LOAD_DIR, 'data', 'energex_network', 'gis')
MB_SHP = os.path.join(LOAD_DIR, 'data', 'MB_2021_AUST_SHP_GDA2020', 'MB_2021_AUST_GDA2020.shp')
DAPR_FILE = os.path.join(LOAD_DIR, '2_dapr_parameters', 'output', 'dapr_parameters_by_bus.csv')
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'output')
MB_TABLE = os.path.join(MATCH_DIR, '1_mesh_blocks', 'output', 'mesh_blocks_QLD.csv')
BUSES_FILE = os.path.join(LOAD_DIR, '1_join_substation_data', 'output', 'site_buses.csv')
OUT = os.path.join(OUTPUT_FOLDER, 'network')
CACHE = os.path.join(OUTPUT_FOLDER, 'cache')
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, LOAD_DIR)
import der_register

sites = pd.read_csv(BUSES_FILE)['SITE'].tolist()
dapr = pd.read_csv(DAPR_FILE)
dapr['SITE'] = dapr['DAPR_NAME'].str.replace(r'\s+Bus\s*\d+$', '', regex=True)
code_to_site = {**dict(zip(dapr['CODE'], dapr['SITE'])), **MANUAL_SITE}
NOT_LOADED = '(substation not in load data)'


def clean(name):
    """'ANN STREET SUBSTATION' -> 'ANN ST', 'ROBINA (SSRBA)' -> 'ROBINA' (to compare with load site names)."""
    s = re.sub(r'\(.*?\)|\bSS[A-Z0-9-]+\b', ' ', str(name).upper())
    s = re.sub(r'\b(ZONE|HV|110KV|132KV|33KV|SUBSTATION|SUB|BULK SUPPLY)\b', ' ', s)
    for a, b in [('MOUNT', 'MT'), ('NORTH ', 'NTH '), ('STREET', 'ST'), ('TERRACE', 'TCE'), ('ISLANDS', 'IS'),
                 ('ISLAND', 'IS'), ('HEADLANDS', 'HEADLAND'), ('ROAD', 'RD')]:
        s = s.replace(a, b)
    return re.sub(r'\s+', ' ', s).strip()


z = gpd.read_file(os.path.join(GIS, 'zone_substations.geojson')).to_crs(CRS_M)
z['CODE'] = z['SITE_REF'].str.replace('^SS', '', regex=True)
z['SITE'] = z['CODE'].map(code_to_site)
z['LINK'] = np.where(z['SITE'].notna(), 'code', '')
z.loc[z['CODE'].isin(MANUAL_SITE), 'LINK'] = 'manual'
by_name = {clean(s): s for s in sites}
name_hit = z['SITE_ALIAS'].map(clean).map(by_name)
coded_sites = set(z['SITE'].dropna())
use_name = z['SITE'].isna() & name_hit.notna() & ~name_hit.isin(coded_sites)
z.loc[use_name, 'SITE'] = name_hit[use_name]
z.loc[use_name, 'LINK'] = 'name'
left_out = z['SITE'].notna() & ~z['SITE'].isin(sites)
z.loc[left_out, 'LINK'] = 'left out in step 1: ' + z.loc[left_out, 'SITE']
z.loc[left_out, 'SITE'] = np.nan

code_point = {}
for i, c in z['CODE'].items():
    code_point.setdefault(c, []).append(i)
site_codes = dapr.groupby('SITE')['CODE'].apply(lambda c: sorted({re.sub(r'\d+$', '', x) for x in c}))
for i, s in z.loc[z['LINK'].isin(['name', 'manual']), 'SITE'].items():
    for c in site_codes.get(s, []):
        code_point.setdefault(c, []).append(i)

print(f"Zone substation points: {len(z)}; linked to a load site: {z['SITE'].notna().sum()} "
      f"({(z['LINK'] == 'code').sum()} by code, {(z['LINK'] == 'name').sum()} by name, "
      f"{(z['LINK'] == 'manual').sum()} manual); load sites with a point: {z['SITE'].nunique()} of {len(sites)}")
print('  load sites without a point:', [s for s in sites if s not in set(z['SITE'])])
z.drop(columns='geometry').to_csv(os.path.join(OUT, 'zone_points.csv'), index=False)

f = pd.concat([gpd.read_file(os.path.join(GIS, n)) for n in ('11kV_HV_Line.geojson', '11kV_HV_cable.geojson')])
f = f.to_crs(CRS_M).dissolve('FEEDER').reset_index()[['FEEDER', 'geometry']]


def candidates(feeder):
    """Zone codes that could be in the feeder name: 'ACR12A' -> ACR; 'AHSAST17' -> AHS, AST."""
    m = re.match(r'^([A-Z]+)', feeder)
    if not m:
        return []
    p = m.group(1)
    return [c for c in dict.fromkeys([p, p[:4], p[:3], p[-3:], p[-4:], p[:2]]) if c in code_point]


rows = []
zg = z.geometry
for r in f.itertuples():
    best = None
    for c in candidates(r.FEEDER):
        for i in code_point[c]:
            d = r.geometry.distance(zg[i])
            if best is None or d < best[1]:
                best = (i, d, 'name')
    if best is None or best[1] > FEEDER_MAX_M:
        d = zg.distance(r.geometry)
        i = d.idxmin()
        if d[i] <= FEEDER_MAX_M:
            best = (i, d[i], 'nearest')
        elif best is not None:
            best = (best[0], best[1], 'name_far')
        else:
            best = (None, d[i], 'none')
    i, d, how = best
    rows.append({'FEEDER': r.FEEDER, 'ZONE_REF': z.at[i, 'SITE_REF'] if i is not None else None,
                 'ZONE_NAME': z.at[i, 'SITE_ALIAS'] if i is not None else None,
                 'SITE': z.at[i, 'SITE'] if i is not None else None, 'HOW': how, 'DIST_M': round(d, 1)})
feeders = pd.DataFrame(rows)
f = f.merge(feeders, on='FEEDER')
print(f"\nFeeders: {len(f)}; matched by name {(f['HOW'] == 'name').sum()} (+{(f['HOW'] == 'name_far').sum()} beyond "
      f"{FEEDER_MAX_M} m), by nearest substation "
      f"{(f['HOW'] == 'nearest').sum()}, unmatched {(f['HOW'] == 'none').sum()}; "
      f"to a load site {f['SITE'].notna().sum()}; median distance to its substation {f['DIST_M'].median():.0f} m")
feeders.to_csv(os.path.join(OUT, 'feeders.csv'), index=False)

t = gpd.read_file(os.path.join(GIS, 'distribution_substations.geojson')).to_crs(CRS_M)
t['KVA'] = pd.to_numeric(t['MAX_KVA'], errors='coerce')
t = gpd.sjoin_nearest(t[['SYSTEM_ID', 'USER_REF_I', 'TR_TYPE', 'KVA', 'geometry']],
                      f[['FEEDER', 'SITE', 'geometry']], how='left', max_distance=TRANSFORMER_MAX_M,
                      distance_col='FEEDER_DIST_M')
t = t[~t.index.duplicated()].drop(columns='index_right')
t['W'] = (t['KVA'].fillna(t.groupby('TR_TYPE')['KVA'].transform('median'))
          .fillna(t['KVA'].median()))
t['SITE'] = t['SITE'].where(t['SITE'].notna() | t['FEEDER'].isna(), NOT_LOADED)
t.loc[t['FEEDER'].isna(), 'SITE'] = '(no feeder within 1 km)'
on_site = t['SITE'].isin(sites)
print(f"\nTransformers: {len(t):,}; on a feeder within {TRANSFORMER_MAX_M} m: {t['FEEDER'].notna().mean():.1%} "
      f"(median {t['FEEDER_DIST_M'].median():.0f} m); supplying a load site: {on_site.mean():.1%} "
      f"({t.loc[on_site, 'W'].sum() / t['W'].sum():.1%} of kVA)")

mb_cache = os.path.join(CACHE, 'mesh_block_polygons_SEQ.pkl')
if os.path.exists(mb_cache):
    mbp = pd.read_pickle(mb_cache)
else:
    print('\nReading Mesh Block polygons (cached after the first run)...')
    mbp = gpd.read_file(MB_SHP, bbox=SEQ_BOX, columns=['MB_CODE21', 'STE_CODE21'])
    mbp = mbp[mbp['STE_CODE21'] == '3'][['MB_CODE21', 'geometry']].rename(columns={'MB_CODE21': 'MB_CODE'})
    mbp.to_pickle(mb_cache)
mbp = mbp.to_crs(CRS_M)
mbt = pd.read_csv(MB_TABLE, dtype={'MB_CODE': str, 'POSTCODE': str})
mbp = mbp.merge(mbt[['MB_CODE', 'POSTCODE', 'DWELLINGS']], on='MB_CODE', how='inner')
print(f"\nMesh Blocks in the SEQ box: {len(mbp):,} ({mbp['DWELLINGS'].sum():,.0f} dwellings)")

t = gpd.sjoin(t, mbp[['MB_CODE', 'POSTCODE', 'geometry']], how='left', predicate='within')
t = t[~t.index.duplicated()].drop(columns='index_right')
t['KVA_MISSING'] = t['KVA'].isna()
t = t.merge(mbt[['MB_CODE', 'MB_CATEGORY']], on='MB_CODE', how='left').set_index(t.index)
t.drop(columns='geometry').to_csv(os.path.join(OUT, 'transformers.csv'), index=False)
ls = t['SITE'].isin(sites)
print(f"Transformers on load sites: {ls.sum():,}; without a kVA rating (median used): {t.loc[ls, 'KVA_MISSING'].sum()}; "
      f"not inside a Mesh Block (no postcode): {t.loc[ls, 'POSTCODE'].isna().sum()}")

inside = t.dropna(subset=['MB_CODE']).groupby(['MB_CODE', 'SITE'])['W'].sum().reset_index()
inside['HOW'] = 'inside'
no_tr = mbp[~mbp['MB_CODE'].isin(inside['MB_CODE']) & (mbp['DWELLINGS'] > 0)]
pts = gpd.GeoDataFrame(no_tr[['MB_CODE']], geometry=no_tr.representative_point(), crs=CRS_M)
near = gpd.sjoin_nearest(pts, t[['SITE', 'geometry']], how='left', max_distance=MESHBLOCK_MAX_M,
                         distance_col='D')
near = near[~near.index.duplicated()].assign(W=1.0, HOW='nearest')
reach = np.clip(1.5 * np.sqrt(no_tr.area), MESHBLOCK_MIN_M, MESHBLOCK_MAX_M)
near.loc[near['D'] > reach.reindex(near.index), 'SITE'] = np.nan
near['SITE'] = near['SITE'].fillna('(outside Energex: no transformer in reach)')
link = pd.concat([inside, near[['MB_CODE', 'SITE', 'W', 'HOW']]], ignore_index=True)
link['FRAC'] = link['W'] / link.groupby('MB_CODE')['W'].transform('sum')
link = link.merge(mbt[['MB_CODE', 'MB_CATEGORY', 'DWELLINGS', 'SAL_NAME', 'POSTCODE']], on='MB_CODE')
link['DWELLINGS_GIVEN'] = link['DWELLINGS'] * link['FRAC']
link.round(4).to_csv(os.path.join(OUT, 'meshblocks.csv'), index=False)
dw = link.groupby('SITE')['DWELLINGS_GIVEN'].sum()
real = dw[dw.index.isin(sites)].sum()
print(f"Dwellings linked to a load site: {real:,.0f} of {dw.sum():,.0f} ({real / dw.sum():.1%}); rest: "
      + ', '.join(f"{k} {v:,.0f}" for k, v in dw.items() if k not in sites))
print(f"  Mesh Blocks with dwellings: {(mbp['DWELLINGS'] > 0).sum():,}; transformer inside: "
      f"{link.loc[link['HOW'] == 'inside', 'MB_CODE'].isin(mbp.loc[mbp['DWELLINGS'] > 0, 'MB_CODE']).sum():,} "
      f"links, nearest transformer: {(link['HOW'] == 'nearest').sum():,}")

buses = pd.read_csv(BUSES_FILE).set_index('SITE')['BUSES']
load_category = dapr.groupby('SITE')['LOAD_CATEGORY'].agg(lambda x: '/'.join(sorted(x.dropna().unique())))
L = link[link['SITE'].isin(sites)]
T = t[t['SITE'].isin(sites)]
chk = pd.DataFrame({
    'BUSES': buses, 'LOAD_CATEGORY': load_category,
    'FEEDERS': feeders[feeders['SITE'].isin(sites)].groupby('SITE').size(),
    'TRANSFORMERS': T.groupby('SITE').size(),
    'KVA': T.groupby('SITE')['W'].sum(),
    'DWELLINGS': L.groupby('SITE')['DWELLINGS_GIVEN'].sum(),
    'SUBURBS': L[L['DWELLINGS_GIVEN'] > 0.5].groupby('SITE')['SAL_NAME'].nunique(),
    'POSTCODES': T.dropna(subset=['POSTCODE']).groupby('SITE')['POSTCODE'].nunique()}).reindex(sites)
kva_type = T.pivot_table(index='SITE', columns='TR_TYPE', values='W', aggfunc='sum').add_prefix('KVA_')
chk = chk.join(kva_type)
chk.round(1).to_csv(os.path.join(OUT, 'site_summary.csv'), index_label='SITE')

pc_dw = mbt.dropna(subset=['POSTCODE']).groupby('POSTCODE')['DWELLINGS'].sum()
on_sites = L.groupby('POSTCODE')['DWELLINGS_GIVEN'].sum()
cov = pd.DataFrame({'DWELLINGS': pc_dw, 'DWELLINGS_ON_LOAD_SITES': on_sites}).fillna(0)
cov = cov[cov.index.isin(link['POSTCODE'].dropna().unique())]
cov['ENERGEX_SHARE'] = cov['DWELLINGS_ON_LOAD_SITES'] / cov['DWELLINGS'].replace(0, np.nan)
tr_on = T.dropna(subset=['POSTCODE']).groupby('POSTCODE').size()
cov['LOAD_SITE_TRANSFORMERS'] = tr_on.reindex(cov.index).fillna(0).astype(int)
cov['OUTSIDE_ENERGEX'] = np.where(cov['DWELLINGS'] > 0, cov['ENERGEX_SHARE'].fillna(0) < OUTSIDE_MAX,
                                  cov['LOAD_SITE_TRANSFORMERS'] == 0)
cov.round(3).to_csv(os.path.join(OUT, 'postcode_coverage.csv'), index_label='POSTCODE')
mixed = cov[cov['OUTSIDE_ENERGEX'] & (cov['DWELLINGS_ON_LOAD_SITES'] > 0)]
print(f"\nPostcodes reached by load-site feeders: {(cov['DWELLINGS_ON_LOAD_SITES'] > 0).sum()}; outside Energex "
      f"(< {OUTSIDE_MAX:.0%} of dwellings on load-site feeders) but partly reached: {' '.join(mixed.index)}")

solar = der_register.der_by_postcode()['solar_kVA']
solar.index = solar.index.astype(str)
dw = L.groupby(['SITE', 'POSTCODE'])['DWELLINGS_GIVEN'].sum()
dw = dw[dw > 0]
sp = dw.rename('DWELLINGS').reset_index()
sp['DWELLING_FRACTION'] = sp['DWELLINGS'] / sp.groupby('POSTCODE')['DWELLINGS'].transform('sum')
sp['POSTCODE_SOLAR_KVA'] = sp['POSTCODE'].map(solar).fillna(0)
sp['SOLAR_KVA'] = sp['POSTCODE_SOLAR_KVA'] * sp['DWELLING_FRACTION']
sp['SOLAR_SHARE'] = sp['SOLAR_KVA'] / sp.groupby('SITE')['SOLAR_KVA'].transform('sum')
sp.round(6).to_csv(os.path.join(OUT, 'site_postcode_solar.csv'), index=False)
print(f"Solar behind load sites: {sp['SOLAR_KVA'].sum() / 1000:,.0f} MVA in {sp['POSTCODE'].nunique()} postcodes; "
      f"sites with no solar: {sorted(set(sites) - set(sp.loc[sp['SOLAR_KVA'] > 0, 'SITE']))}")
print('Saved to', OUT)

### STEP 3.4 - Voronoi methods: draw each zone substation's service area from its distribution transformers, then
### overlay the areas with the postcode boundaries. Mesh Blocks are NOT used in this step.

import os
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

warnings.filterwarnings('ignore', category=UserWarning)

CRS_M = 7856
SEQ_BOX = (151.8, -28.45, 153.65, -25.75)
ZOOM_BOX = (152.98, -27.53, 153.08, -27.45)
CLIP_BUFFER_M = 2000
NOT_LOADED = '(substation not in load data)'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MATCH_DIR = os.path.dirname(SCRIPT_DIR)
LOAD_DIR = os.path.dirname(MATCH_DIR)
GIS = os.path.join(LOAD_DIR, 'data', 'energex_network', 'gis')
POA_SHP = os.path.join(LOAD_DIR, 'data', 'POA_2021_AUST_GDA2020_SHP', 'POA_2021_AUST_GDA2020.shp')
NET = os.path.join(MATCH_DIR, '2_network_matching', 'output', 'network')
SOLAR_FILE = os.path.join(NET, 'site_postcode_solar.csv')
sys.path.insert(0, MATCH_DIR)
sys.dont_write_bytecode = True
from split_tools import solar_shares, BUSINESS, BUSINESS_TYPES
U_FILE = os.path.join(MATCH_DIR, '3_mesh_block_split', 'output', 'utilisation_factors.csv')
SITE_LOAD_FILE = os.path.join(os.path.dirname(MATCH_DIR), '1_join_substation_data', 'output', 'load_MW_by_site.csv')
OUT = os.path.join(SCRIPT_DIR, 'output')
CACHE = os.path.join(OUT, 'cache')
os.makedirs(CACHE, exist_ok=True)

site_load = pd.read_csv(SITE_LOAD_FILE, index_col='DATETIME', parse_dates=True)
sites = site_load.columns.tolist()
summary = pd.read_csv(os.path.join(NET, 'site_summary.csv')).set_index('SITE')

tr = gpd.read_file(os.path.join(GIS, 'distribution_substations.geojson'), columns=['SYSTEM_ID']).to_crs(CRS_M)
tr['SYSTEM_ID'] = tr['SYSTEM_ID'].astype(str)
tl = pd.read_csv(os.path.join(NET, 'transformers.csv'), dtype={'SYSTEM_ID': str})
u = pd.read_csv(U_FILE, index_col=0)['U_MW_PER_MVA']
tl['WEIGHT'] = tl['W'] * tl['TR_TYPE'].map(u).fillna(0)
tl['BUSINESS'] = tl['MB_CATEGORY'].isin(BUSINESS) | tl['TR_TYPE'].isin(BUSINESS_TYPES)
tl['KVA_W'] = tl['W']
tr = tr.merge(tl[['SYSTEM_ID', 'SITE', 'WEIGHT', 'KVA_W', 'BUSINESS']], on='SYSTEM_ID', how='left')
tr['SITE'] = tr['SITE'].where(tr['SITE'].isin(sites), NOT_LOADED)
tr['WEIGHT'] = tr['WEIGHT'].fillna(0)
tr['KVA_W'] = tr['KVA_W'].fillna(0)
tr['BUSINESS'] = tr['BUSINESS'].fillna(False).astype(bool)
tr['X'], tr['Y'] = tr.geometry.x, tr.geometry.y
tr = tr.sort_values('WEIGHT', ascending=False)
seeds = tr.groupby(['X', 'Y'], sort=False).agg(SYSTEM_ID=('SYSTEM_ID', 'first'), SITE=('SITE', 'first'),
                                               BUSINESS=('BUSINESS', 'first'), WEIGHT=('WEIGHT', 'sum'),
                                               KVA_W=('KVA_W', 'sum'),
                                               TRANSFORMERS=('SYSTEM_ID', 'size')).reset_index()
seeds = gpd.GeoDataFrame(seeds, geometry=gpd.points_from_xy(seeds['X'], seeds['Y']), crs=CRS_M)
print(f"Transformers: {len(tr):,} ({(tr['SITE'] != NOT_LOADED).sum():,} on load sites) -> {len(seeds):,} seed "
      f"locations ({(seeds['TRANSFORMERS'] > 1).sum()} with more than one transformer)")

cov = pd.read_csv(os.path.join(NET, 'postcode_coverage.csv'), dtype={'POSTCODE': str})
reached = set(cov.loc[(cov['DWELLINGS_ON_LOAD_SITES'] > 0) | (cov['LOAD_SITE_TRANSFORMERS'] > 0), 'POSTCODE'])
poa = gpd.read_file(POA_SHP, bbox=SEQ_BOX).rename(columns={'POA_CODE21': 'POSTCODE'})
poa = poa[poa['POSTCODE'].isin(reached) & poa.geometry.notna()][['POSTCODE', 'geometry']].to_crs(CRS_M)
mask_cache = os.path.join(CACHE, 'clip_mask.pkl')
if os.path.exists(mask_cache):
    mask = pd.read_pickle(mask_cache)
else:
    grid = np.unique(np.round(shapely.get_coordinates(tr.geometry.values) / 500) * 500, axis=0)
    near = shapely.union_all(shapely.buffer(shapely.points(grid), CLIP_BUFFER_M, quad_segs=4))
    mask = shapely.intersection(shapely.union_all(poa.geometry.values), near)
    pd.to_pickle(mask, mask_cache)
shapely.prepare(mask)
print(f"Clip area: {mask.area / 1e6:,.0f} km2 in {len(poa)} postcodes reached by load-site feeders")

cells = shapely.get_parts(shapely.voronoi_polygons(shapely.multipoints(seeds.geometry.values),
                                                    extend_to=mask.envelope))
cells = gpd.GeoDataFrame(geometry=cells, crs=CRS_M)
cells = gpd.sjoin(cells, seeds[['SYSTEM_ID', 'SITE', 'BUSINESS', 'WEIGHT', 'KVA_W', 'TRANSFORMERS', 'geometry']],
                  predicate='contains', how='inner').drop(columns='index_right').reset_index(drop=True)
g = cells.geometry.values.copy()
edge = ~shapely.contains_properly(mask, g)
g[edge] = shapely.intersection(g[edge], mask)
cells['geometry'] = g
cells = cells[~cells.geometry.is_empty].copy()
cells['CELL_KM2'] = cells.geometry.area / 1e6
print(f"Voronoi cells: {len(cells):,}; median {cells['CELL_KM2'].median() * 1e6:,.0f} m2 "
      f"({cells.loc[cells['SITE'] != NOT_LOADED, 'CELL_KM2'].median() * 1e6:,.0f} m2 on load sites)")

merged = cells.groupby('SITE').geometry.apply(lambda x: shapely.coverage_union_all(x.values))
areas = gpd.GeoDataFrame({'SITE': merged.index}, geometry=list(merged.values), crs=CRS_M)

pieces = gpd.overlay(cells[cells['SITE'] != NOT_LOADED], poa, how='intersection', keep_geom_type=True)
pieces['KM2'] = pieces.geometry.area / 1e6
pieces = pieces[pieces['KM2'] > 1e-6]
pieces['CELL_FRACTION'] = pieces['KM2'] / pieces.groupby('SYSTEM_ID')['KM2'].transform('sum')
pieces['W'] = pieces['WEIGHT'] * pieces['CELL_FRACTION']
pieces['W_KVA'] = pieces['KVA_W'] * pieces['CELL_FRACTION']
crossing = (pieces.groupby('SYSTEM_ID')['POSTCODE'].nunique() > 1).sum()
print(f"Overlay: {len(pieces):,} cell-postcode pieces; {crossing:,} cells cross a postcode boundary")


def write(name, col):
    """Site -> postcode shares from the pieces, weighted by column col (KM2 or W)."""
    p = pieces.assign(V=pieces[col], VB=pieces[col] * pieces['BUSINESS'])
    m = p.groupby(['SITE', 'POSTCODE']).agg(AREA_KM2=('KM2', 'sum'), CELLS=('SYSTEM_ID', 'nunique'),
                                            WEIGHT=('V', 'sum'), BUS=('VB', 'sum')).reset_index()
    m = m[m['WEIGHT'] > 0]
    m['SHARE'] = m['WEIGHT'] / m.groupby('SITE')['WEIGHT'].transform('sum')
    m['POSTCODE_FRACTION'] = m['WEIGHT'] / m.groupby('POSTCODE')['WEIGHT'].transform('sum')
    m['BUSINESS_FRACTION'] = m['BUS'] / m['WEIGHT']
    m['OTHER_FRACTION'] = 0.0
    m['BUSES'] = m['SITE'].map(summary['BUSES'])
    m['LOAD_CATEGORY'] = m['SITE'].map(summary['LOAD_CATEGORY'])
    m['SITES_IN_POSTCODE'] = m.groupby('POSTCODE')['SITE'].transform('nunique')
    m['SOLAR_SHARE'] = solar_shares(m, SOLAR_FILE)
    assert ((m.groupby('SITE')['SHARE'].sum() - 1).abs() < 1e-9).all()
    folder = os.path.join(OUT, name)
    os.makedirs(folder, exist_ok=True)
    m[['SITE', 'BUSES', 'LOAD_CATEGORY', 'POSTCODE', 'SHARE', 'SOLAR_SHARE', 'POSTCODE_FRACTION', 'AREA_KM2', 'CELLS', 'WEIGHT',
       'BUSINESS_FRACTION', 'OTHER_FRACTION', 'SITES_IN_POSTCODE']].round(6).to_csv(
        os.path.join(folder, 'site_postcode_mapping.csv'), index=False)
    print(f"{name:17s} {m['SITE'].nunique()} sites -> {m['POSTCODE'].nunique()} postcodes, {len(m)} pairs; "
          f"missing sites: {sorted(set(sites) - set(m['SITE'])) or 'none'}")
    return m


maps = {'voronoi_equal': write('voronoi_equal', 'KM2'), 'voronoi_kva': write('voronoi_kva', 'W_KVA'),
        'voronoi_weighted': write('voronoi_weighted', 'W')}

a = areas[areas['SITE'] != NOT_LOADED].copy()
a['AREA_KM2'] = (a.geometry.area / 1e6).round(2)
a['POSTCODES'] = a['SITE'].map(maps['voronoi_equal'].groupby('SITE')['POSTCODE'].apply(lambda x: ' '.join(sorted(x))))
a['geometry'] = a.geometry.simplify(20)
a.to_crs(4326).to_file(os.path.join(OUT, 'service_areas.geojson'), driver='GeoJSON')
pts = gpd.GeoDataFrame(poa[['POSTCODE']], geometry=poa.representative_point(), crs=CRS_M)
pj = gpd.sjoin(pts, areas, predicate='within', how='left')[['POSTCODE', 'SITE']]
pj.to_csv(os.path.join(OUT, 'postcode_to_site.csv'), index=False)
print(f"Service areas: {len(a)} load sites, median {a['AREA_KM2'].median():.1f} km2; many-to-one lookup: "
      f"{pj['SITE'].isin(sites).sum()} of {len(pj)} postcodes in a load-site area, "
      f"{len(set(sites) - set(pj['SITE']))} sites given no postcode")

W4326 = 4326
INK, MUTED, GRID, SURFACE, NODATA = '#0b0b0b', '#52514e', '#c9c8c3', '#fcfcfb', '#e6e5e0'
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
             '#1c5cab', '#184f95', '#104281', '#0d366b']

la = areas[areas['SITE'] != NOT_LOADED].reset_index(drop=True)
nb = gpd.sjoin(la[['SITE', 'geometry']], la[['SITE', 'geometry']].assign(geometry=la.buffer(50)),
               predicate='intersects')
nb = nb[nb['SITE_left'] != nb['SITE_right']].groupby('SITE_left')['SITE_right'].apply(set)
colour_of = {}
for s in la.assign(A=la.area).sort_values('A', ascending=False)['SITE']:
    used = {colour_of.get(n) for n in nb.get(s, set())}
    colour_of[s] = next((c for c in CATEGORICAL if c not in used), CATEGORICAL[len(colour_of) % len(CATEGORICAL)])

aw = areas.to_crs(W4326)
law = aw[aw['SITE'] != NOT_LOADED]
poa_w = poa.to_crs(W4326)
zone = gpd.read_file(os.path.join(GIS, 'zone_substations.geojson')).to_crs(W4326)
zp = pd.read_csv(os.path.join(NET, 'zone_points.csv'))
zone['SITE'] = zone['SITE_REF'].map(zp.set_index('SITE_REF')['SITE'])
zone = zone[zone['SITE'].isin(sites)]
zx = gpd.GeoSeries([shapely.box(*ZOOM_BOX)], crs=W4326).to_crs(CRS_M).iloc[0]
cz = cells[cells.intersects(zx.buffer(500))].to_crs(W4326)
czl = cz[cz['SITE'] != NOT_LOADED]


def frame(ax, box, title):
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3])
    ax.set_aspect(1 / 0.885)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(title, loc='left', fontsize=11, color=INK)


def zoom_outline(ax):
    ax.add_patch(plt.Rectangle(ZOOM_BOX[:2], ZOOM_BOX[2] - ZOOM_BOX[0], ZOOM_BOX[3] - ZOOM_BOX[1], fill=False,
                               edgecolor=INK, lw=1.2, ls='--', zorder=10))
    ax.text(ZOOM_BOX[2] + 0.04, ZOOM_BOX[3] + 0.04, 'zoom (b)', fontsize=9, color=INK, zorder=11,
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))


fig, axes = plt.subplots(1, 2, figsize=(16, 9.5), gridspec_kw={'width_ratios': [1, 1.15]}, facecolor=SURFACE)
ax = axes[0]
aw[aw['SITE'] == NOT_LOADED].plot(ax=ax, color=NODATA, edgecolor='none')
law.plot(ax=ax, color=[colour_of[s] for s in law['SITE']], alpha=0.8, edgecolor='white', linewidth=0.3)
poa_w.boundary.plot(ax=ax, color=INK, linewidth=0.2)
frame(ax, SEQ_BOX, f'(a) {len(law)} zone substation service areas, South East Queensland')
zoom_outline(ax)
ax = axes[1]
cz[cz['SITE'] == NOT_LOADED].plot(ax=ax, color=NODATA, edgecolor='white', linewidth=0.3)
czl.plot(ax=ax, color=[colour_of[s] for s in czl['SITE']], alpha=0.55, edgecolor='white', linewidth=0.4)
law.boundary.plot(ax=ax, color=INK, linewidth=1.2)
poa_w.boundary.plot(ax=ax, color=MUTED, linewidth=0.9, linestyle='--')
mx, my = 0.08 * (ZOOM_BOX[2] - ZOOM_BOX[0]), 0.03 * (ZOOM_BOX[3] - ZOOM_BOX[1])
zin = zone[zone.geometry.x.between(ZOOM_BOX[0], ZOOM_BOX[2] - mx) & zone.geometry.y.between(ZOOM_BOX[1] + my,
                                                                                             ZOOM_BOX[3] - my)]
zin.plot(ax=ax, marker='^', color=INK, markersize=50, zorder=5)
for _, r in zin.iterrows():
    ax.annotate(r['SITE'], (r.geometry.x, r.geometry.y), xytext=(5, 4), textcoords='offset points', fontsize=8,
                color=INK, zorder=6, bbox=dict(facecolor=SURFACE, edgecolor='none', alpha=0.8, pad=1))
frame(ax, ZOOM_BOX, '(b) Inner Brisbane: one Voronoi cell per distribution transformer, merged by substation')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=CATEGORICAL[0], alpha=0.8, label='Service area of one zone substation '
                                                                       '(colours only separate neighbours)'),
                    Line2D([], [], color=INK, lw=1.2, label='Service area boundary'),
                    Line2D([], [], color=MUTED, lw=0.9, ls='--', label='Postcode (ABS postal area) boundary'),
                    Line2D([], [], marker='^', ls='', color=INK, markersize=7, label='Zone substation (b)'),
                    Patch(facecolor=NODATA, label='Substation not in the load data')],
           loc='lower center', ncol=5, frameon=False, fontsize=9.5)
fig.suptitle('Zone substation service areas from transformer Voronoi cells (after Day et al. 2024)', x=0.01,
             ha='left', fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(os.path.join(OUT, 'fig_1_service_areas.png'), dpi=220, facecolor=SURFACE)
plt.close(fig)

from matplotlib.colors import ListedColormap
cmap = ListedColormap(BLUE_RAMP)
cl = cells[cells['SITE'] != NOT_LOADED].to_crs(W4326)
dens = cl['WEIGHT'] / cl['CELL_KM2']
norm = LogNorm(vmin=max(dens.quantile(0.05), 1), vmax=dens.quantile(0.98))
dz = czl['WEIGHT'] / czl['CELL_KM2']
fig, axes = plt.subplots(1, 2, figsize=(16, 9.5), gridspec_kw={'width_ratios': [1, 1.15]}, facecolor=SURFACE)
ax = axes[0]
aw[aw['SITE'] == NOT_LOADED].plot(ax=ax, color=NODATA, edgecolor='none')
cl.assign(D=dens.clip(lower=norm.vmin)).plot(ax=ax, column='D', cmap=cmap, norm=norm, edgecolor='none')
poa_w.boundary.plot(ax=ax, color=INK, linewidth=0.2)
frame(ax, SEQ_BOX, '(a) Load weight per km² of each transformer cell, South East Queensland')
zoom_outline(ax)
ax = axes[1]
czl.assign(D=dz.clip(lower=norm.vmin)).plot(ax=ax, column='D', cmap=cmap, norm=norm, edgecolor='white',
                                            linewidth=0.3)
cz[cz['SITE'] == NOT_LOADED].plot(ax=ax, color=NODATA, edgecolor='white', linewidth=0.3)
poa_w.boundary.plot(ax=ax, color=INK, linewidth=1.2)
rp = poa_w.representative_point()
for (_, r), p in zip(poa_w.iterrows(), rp):
    if ZOOM_BOX[0] < p.x < ZOOM_BOX[2] and ZOOM_BOX[1] < p.y < ZOOM_BOX[3]:
        ax.text(p.x, p.y, r['POSTCODE'], fontsize=8, color=INK, ha='center', va='center', zorder=6,
                bbox=dict(facecolor=SURFACE, edgecolor='none', alpha=0.8, pad=1))
frame(ax, ZOOM_BOX, '(b) Inner Brisbane: cells split by area at postcode boundaries (black)')
fig.tight_layout(rect=(0, 0.09, 1, 0.96))
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, cax=fig.add_axes([0.25, 0.05, 0.5, 0.018]), orientation='horizontal')
cb.set_label('Average load weight: transformer kVA × utilisation of its type, per km² of its cell '
             '(kW/km², log scale)', color=INK)
cb.outline.set_edgecolor(GRID)
fig.suptitle('voronoi_weighted: each cell carries its transformer\'s kVA × utilisation, spread evenly over the '
             'cell', x=0.01, ha='left', fontsize=13, color=INK)
fig.savefig(os.path.join(OUT, 'fig_2_voronoi_weighted.png'), dpi=220, facecolor=SURFACE)
plt.close(fig)
print('Saved to', OUT)

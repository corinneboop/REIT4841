### STEP 2 - Build the analysis table: one row per Energex postcode with every outcome and variable, defined once.

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SOCIAL_DIR = os.path.dirname(HERE)
CODE = os.path.dirname(SOCIAL_DIR)
sys.path.insert(0, SOCIAL_DIR)
from variable_sets import SOCIAL, NON_OVERLAPPING

OUT = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)
EXCESS_DAILY = os.path.join(CODE, '2_load_data', '4_reverse_flow', 'output', 'reverse_flow_daily_by_postcode.csv')
COVERAGE = os.path.join(CODE, '2_load_data', '3_postcode_matching', '2_network_matching', 'output', 'network',
                        'postcode_coverage.csv')
CAPACITY = os.path.join(CODE, '2_load_data', '5_frequency_magnitude', 'output', 'capacity_by_postcode.csv')
SOCIAL_FILE = os.path.join(SOCIAL_DIR, '1_social_variables', 'output', 'social_variables.csv')
sys.path.insert(0, os.path.join(CODE, '2_load_data'))
import der_register as derq
POSTCODE_FILE = os.path.join(CODE, '2_load_data', 'data', 'raw_postcodes.csv')
MATCH = os.path.join(CODE, '2_load_data', '3_postcode_matching')
REPORTED = os.path.join(MATCH, '5_compare_methods', 'output', 'energy_check_by_postcode.csv')
MAPPING = os.path.join(MATCH, '5_compare_methods', 'output', 'selected', 'site_postcode_mapping.csv')
SITE_LOAD = os.path.join(CODE, '2_load_data', '1_join_substation_data', 'output', 'load_MW_by_site.csv')
USAGE_FILE = os.path.join(MATCH, 'data', 'Energy-Usage-Postcode-31-Mar-2026.xlsx')
MB_FILE = os.path.join(MATCH, '1_mesh_blocks', 'output', 'mesh_blocks_QLD.csv')
TRANSFORMERS = os.path.join(MATCH, '2_network_matching', 'output', 'network', 'transformers.csv')
DAPR_PARAMS = os.path.join(CODE, '2_load_data', '2_dapr_parameters', 'output', 'dapr_parameters_by_bus.csv')
DAPR_VALID = os.path.join(CODE, '2_load_data', '2_dapr_parameters', 'output', 'dapr_validation.csv')
POA_SHP = os.path.join(CODE, '2_load_data', 'data', 'POA_2021_AUST_GDA2020_SHP', 'POA_2021_AUST_GDA2020.shp')
USAGE_MONTHS = pd.period_range('2025-07', '2026-03', freq='M')
REGIONS = {'Brisbane': 'Brisbane', 'Gold Coast': 'Gold Coast', 'Sunshine Coast': 'Sunshine Coast',
           'Logan - Beaudesert': 'Logan - Beaudesert', 'Moreton Bay': 'Moreton Bay', 'Ipswich': 'Ipswich'}
MIN_SOLAR_SITES = 100
ALPHA = 0.05
Y = 'log_excess'

LABELS = {
    'log_income': 'Income: household (log median)',
    'pct_age_20_29': 'Age: % 20-29', 'pct_over_65': 'Age: % 65+',
    'pct_families_children': 'Family: % families with children under 15',
    'pct_owned_outright': 'Tenure: % owned outright', 'pct_owned_mortgage': 'Tenure: % owned with mortgage',
    'pct_rented': 'Tenure: % rented', 'pct_house': 'Dwelling: % separate house', 'pct_apartment': 'Dwelling: % apartment',
    'pct_married': 'Relationship: % married', 'pct_bachelor_or_higher': 'Education: % bachelor or higher',
    'pct_trades_workers': 'Occupation: % technicians and trades workers', 'pct_fulltime_15plus': 'Employment: % full time',
    'pct_unemployed_15plus': 'Employment: % unemployed', 'pct_not_in_labour_force': 'Employment: % not in labour force',
    'business_share_pct': 'Load: % business consumption',
    'res_kWh_per_customer_day': 'Load: residential kWh per customer per day',
    'log_bus_customers_per_1000_dw': 'Load: business customers per 1,000 dwellings (log)',
    'industrial_substation_pct': 'Load: % energy from industrial substations',
    'log_dwelling_density': 'Built form: dwellings per km2 residential land (log)',
    'res_tr_kVA_per_dwelling': 'Network: residential transformer kVA per dwelling',
    'log_ceg_per_solar_kVA': 'Network: substation CEG per kVA of solar (log)',
    'dist_coast_km': 'Location: distance to coast (km)',
    'pv_kVA_per_dwelling': 'Solar: kVA per dwelling', 'avg_solar_kVA_per_site': 'Solar: kVA per solar site',
    'major_city': 'Location: major city (1/0)'}
THEME = {('log_income' if v == 'median_hhd_income_weekly' else v): t for t, vs in NON_OVERLAPPING.items() for v in vs}
THEME.update({'business_share_pct': 'Load composition', 'res_kWh_per_customer_day': 'Load composition',
              'log_bus_customers_per_1000_dw': 'Load composition', 'industrial_substation_pct': 'Load composition',
              'log_dwelling_density': 'Built form / network',
              'res_tr_kVA_per_dwelling': 'Built form / network', 'log_ceg_per_solar_kVA': 'Built form / network',
              'pv_kVA_per_dwelling': 'Technical', 'avg_solar_kVA_per_site': 'Technical', 'major_city': 'Location',
              'dist_coast_km': 'Location'})
CONTEXT = ['business_share_pct', 'res_kWh_per_customer_day', 'log_bus_customers_per_1000_dw',
           'industrial_substation_pct', 'log_dwelling_density', 'res_tr_kVA_per_dwelling', 'log_ceg_per_solar_kVA',
           'pv_kVA_per_dwelling', 'avg_solar_kVA_per_site', 'major_city', 'dist_coast_km']
X_ALL = ['log_income' if v == 'median_hhd_income_weekly' else v for v in SOCIAL] + \
        CONTEXT

daily = pd.read_csv(EXCESS_DAILY, dtype={'POSTCODE': str})
use = daily[~daily['DAY_FLAGGED'].astype(bool)]
ex = use.groupby('POSTCODE').agg(days_used=('DATE', 'size'), excess_MWh=('REVERSE_TRAP_MWh', 'sum'),
                                 days_with_excess=('REVERSE_TRAP_MWh', lambda x: (x > 0).sum()))
ex['annual_excess_MWh'] = ex['excess_MWh'] * 365 / ex['days_used']
cov = pd.read_csv(COVERAGE, dtype={'POSTCODE': str}).set_index('POSTCODE')
allseg = derq.mean_over(derq.SOLAR_FIELDS, derq.SOLAR_QUARTERS)
res = derq.mean_over(derq.SOLAR_FIELDS, derq.SOLAR_QUARTERS, 'res').rename(columns=lambda c: 'res_' + c)
resb = derq.mean_over(['battery_sites', 'battery_kWh'], derq.ALL_QUARTERS, 'res')
allb = derq.mean_over(['battery_sites', 'battery_kWh'], derq.ALL_QUARTERS).rename(columns=lambda c: c + '_all')
qres = derq.quarter_table('battery_sites', 'res')
added = (qres['2026-03'] - qres['2025-07']).rename('battery_sites_added')
social = pd.read_csv(SOCIAL_FILE, dtype={'POSTCODE': str}).set_index('POSTCODE')
pc = pd.read_csv(POSTCODE_FILE, dtype={'postcode': str}, low_memory=False)
pc = pc[(pc['state'] == 'QLD') & (pc['type'] == 'Delivery Area')]
geo = pc.groupby('postcode').agg(lat=('lat', 'median'), long=('long', 'median'),
                                 RA=('RA_2021_NAME', lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
                                 SA4=('sa4name', lambda s: s.mode().iloc[0] if s.notna().any() else np.nan))
rep_e = pd.read_csv(REPORTED, dtype={'POSTCODE': str}).set_index('POSTCODE')
mp = pd.read_csv(MAPPING, dtype={'POSTCODE': str})
mp['E'] = mp['SITE'].map(pd.read_csv(SITE_LOAD, index_col=0).mean()) * mp['SHARE']
main_site = mp.sort_values('E').groupby('POSTCODE')['SITE'].last()

d = ex.join(allseg).join(res).join(social).join(geo)
d['outside_energex'] = cov['OUTSIDE_ENERGEX'].reindex(d.index).fillna(True).astype(bool)
n0 = len(d)
d = d[~d['outside_energex'] & (d['res_solar_sites'] >= MIN_SOLAR_SITES)].copy()
d['excess_kWh_per_site'] = d['annual_excess_MWh'] * 1000 / d['solar_sites']
d[Y] = np.log1p(d['excess_kWh_per_site'])
d['has_excess'] = (d['excess_kWh_per_site'] > 0).astype(int)
d['log_income'] = np.log(d['median_hhd_income_weekly'])
d['pv_kVA_per_dwelling'] = d['solar_kVA'] / d['dwellings']
d['avg_solar_kVA_per_site'] = d['res_solar_kVA'] / d['res_solar_sites']
d['major_city'] = (d['RA'] == 'Major Cities of Australia').astype(int)
d['business_share_pct'] = 100 * rep_e['REPORTED_BUSINESS_SHARE'].reindex(d.index)
d['region'] = d['SA4'].map(lambda x: next((v for k, v in REGIONS.items() if isinstance(x, str) and k in x), 'Other SEQ'))
d['main_substation'] = main_site.reindex(d.index)

u = pd.read_excel(USAGE_FILE)
u = u.melt(id_vars=['Post Code', 'Customer Consumption Type', 'Year', 'Measure Type'],
           value_vars=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                       'October', 'November', 'December'], var_name='M', value_name='v')
u['MONTH'] = pd.to_datetime(u['Year'].astype(str) + '-' + u['M'], format='%Y-%B').dt.to_period('M')
u = u[u['MONTH'].isin(USAGE_MONTHS) & (u['v'] > 0)]
u['POSTCODE'] = u['Post Code'].astype(int).astype(str).str.zfill(4)
um = u.groupby(['POSTCODE', 'Customer Consumption Type', 'Measure Type'])['v'].mean().unstack([1, 2])
d['res_kWh_per_customer_day'] = um[('Residential', 'Energy per customer per day (kWh)')].reindex(d.index)
d['bus_customers_per_1000_dw'] = 1000 * um[('Business', 'Customer Count')].reindex(d.index) / d['dwellings']
mp['ind'] = mp['LOAD_CATEGORY'].fillna('').str.contains('Industrial|Continuous')
d['industrial_substation_pct'] = (100 * mp.assign(x=mp['E'] * mp['ind']).groupby('POSTCODE')['x'].sum()
                                  / mp.groupby('POSTCODE')['E'].sum()).reindex(d.index)
mb = pd.read_csv(MB_FILE, dtype={'POSTCODE': str})
mbr = mb[mb['MB_CATEGORY'] == 'Residential'].groupby('POSTCODE')[['DWELLINGS', 'AREA_SQKM']].sum()
d['dwelling_density'] = (mbr['DWELLINGS'] / mbr['AREA_SQKM']).reindex(d.index)
tr = pd.read_csv(TRANSFORMERS, dtype={'POSTCODE': str})
trr = tr[(tr['MB_CATEGORY'] == 'Residential') & tr['TR_TYPE'].isin(['POLE', 'PADMOUNT'])]
d['res_tr_kVA_per_dwelling'] = (trr.groupby('POSTCODE')['W'].sum().reindex(d.index) / d['dwellings'])
dp = pd.read_csv(DAPR_PARAMS)[['CODE', 'CAPACITY_FOR_EMBEDDED_GENERATION_MVA__S2526']]
dp = dp.merge(pd.read_csv(DAPR_VALID)[['CODE', 'SITE']], on='CODE')
ceg = dp.groupby('SITE')['CAPACITY_FOR_EMBEDDED_GENERATION_MVA__S2526'].sum(min_count=1) * 1000
pv_pc = allseg['solar_kVA']
site_pv = (mp.assign(pv=mp['POSTCODE'].map(pv_pc) * mp['POSTCODE_FRACTION'])).groupby('SITE')['pv'].sum()
mp['ceg_ratio'] = mp['SITE'].map(ceg / site_pv.replace(0, np.nan))
okc = mp.dropna(subset=['ceg_ratio'])
d['ceg_kVA_per_solar_kVA'] = (okc.assign(x=okc['E'] * okc['ceg_ratio']).groupby('POSTCODE')['x'].sum()
                              / okc.groupby('POSTCODE')['E'].sum()).reindex(d.index)
import geopandas as gpd
import shapely
poa = gpd.read_file(POA_SHP, bbox=(150.0, -30.5, 154.0, -24.0))
poa = poa[poa.geometry.notna()].to_crs(7856)
land = shapely.union_all(shapely.simplify(poa.geometry.values, 50))
land = land.buffer(1000).buffer(-1000)
parts = list(land.geoms) if land.geom_type == 'MultiPolygon' else [land]
mainland = max(parts, key=lambda g: g.area)
coast = shapely.Polygon(mainland.exterior).exterior
pts = gpd.GeoSeries(gpd.points_from_xy(d['long'], d['lat']), crs=4326).to_crs(7856)
d['dist_coast_km'] = [p.distance(coast) / 1000 for p in pts.values]
d['log_dwelling_density'] = np.log(d['dwelling_density'])
d['log_bus_customers_per_1000_dw'] = np.log(d['bus_customers_per_1000_dw'])
d['log_ceg_per_solar_kVA'] = np.log(d['ceg_kVA_per_solar_kVA'])
d = d.join(resb).join(allb).join(added)
d['uptake_pct'] = 100 * d['battery_sites'] / d['res_solar_sites']
d['battery_kWh_per_solar_site'] = d['battery_kWh'] / d['res_solar_sites']
d['log_battery_kWh_per_solar_site'] = np.log(d['battery_kWh_per_solar_site'].where(d['battery_kWh_per_solar_site'] > 0))
d['kWh_per_battery_site'] = np.where(d['battery_sites'] > 0, d['battery_kWh'] / d['battery_sites'], np.nan)
d['uptake_all_pct'] = 100 * d['battery_sites_all'] / d['solar_sites']
d['battery_kWh_per_solar_site_all'] = d['battery_kWh_all'] / d['solar_sites']
d['new_uptake_pp'] = 100 * d['battery_sites_added'] / d['res_solar_sites']
dk = use.assign(kWh=use['REVERSE_TRAP_MWh'] * 1000).groupby('POSTCODE')['kWh']
for lab, qv in [('p90', 0.9), ('p75', 0.75), ('median', 0.5)]:
    need = dk.quantile(qv).reindex(d.index)
    d[f'need_{lab}_kWh'] = need
    d[f'need_{lab}_kWh_per_site'] = need / d['solar_sites']
    d[f'coverage_{lab}'] = (d['battery_kWh_all'] / need).where(need > 0)
d['log_need_p90_per_site'] = np.log1p(d['need_p90_kWh_per_site'])
d['has_need_p90'] = (d['need_p90_kWh_per_site'] > 0).astype(int)
d['energy_coverage'] = d['coverage_p90']
d['pct_days_with_excess'] = 100 * d['days_with_excess'] / d['days_used']
cp = pd.read_csv(CAPACITY, dtype={'POSTCODE': str}).set_index('POSTCODE').reindex(d.index)
d['network_status'] = cp['STATUS']
d['network_status_detail'] = cp['STATUS_DETAIL']
d['network_status_n1'] = cp['STATUS_N1']
d['above_capacity'] = (cp['STATUS'] == 'above capacity').astype(int)
d['at_or_near_capacity'] = cp['STATUS'].isin(['above capacity', 'near capacity']).astype(int)
d['pct_reverse_above_ceg'] = cp['PCT_ABOVE_CEG']
d['above_ceg_kWh_per_site'] = cp['ABOVE_kWh_PER_SOLAR_SITE']
d['within_ceg_kWh_per_site'] = cp['WITHIN_kWh_PER_SOLAR_SITE']
d['above_ceg_n1_kWh_per_site'] = cp['ABOVE_N1_kWh_PER_SOLAR_SITE']
d['log_above_ceg_per_site'] = np.log1p(d['above_ceg_kWh_per_site'])
d['main_reverse_site'] = cp['MAIN_REVERSE_SITE']
d['p90_pct_of_ceg'] = cp['P90_PEAK_PCT_OF_CEG']
d['p90_pct_of_ceg_n1'] = cp['P90_PEAK_PCT_OF_CEG_N1']
d['ceg_n1_MW'] = cp['CEG_N1_SUMMER_MW']
d['capacity_weight'] = (d['p90_pct_of_ceg'] / 100).clip(0, 1)
d.loc[d['excess_kWh_per_site'] == 0, 'capacity_weight'] = d.loc[d['excess_kWh_per_site'] == 0, 'capacity_weight'].fillna(0)
d['excess_weighted_kWh_per_site'] = d['excess_kWh_per_site'] * d['capacity_weight']
d['log_excess_weighted'] = np.log1p(d['excess_weighted_kWh_per_site'])
d['need_p90_weighted_kWh_per_site'] = d['need_p90_kWh_per_site'] * d['capacity_weight']
d['log_need_p90_weighted'] = np.log1p(d['need_p90_weighted_kWh_per_site'])
for g in sorted(x for x in d['region'].dropna().unique() if x != 'Brisbane'):
    d['region_' + g] = (d['region'] == g).astype(float)
d = d.dropna(subset=[Y] + X_ALL)
d.to_csv(os.path.join(OUT, 'analysis_dataset.csv'), index_label='POSTCODE')
print(f"Sample A (Energex) - postcodes with excess results: {n0}; Energex, >= {MIN_SOLAR_SITES} residential solar sites and census data: {len(d)}")
print(f"Excess per solar site: median {d['excess_kWh_per_site'].median():.0f} kWh/yr, mean {d['excess_kWh_per_site'].mean():.0f}; "
      f"{(d['has_excess'] == 0).sum()} postcodes with none; skew raw {d['excess_kWh_per_site'].skew():.1f}, log {d[Y].skew():.1f}")

desc = d[['annual_excess_MWh', 'excess_kWh_per_site', Y] + X_ALL].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
desc.insert(0, 'description', ['Annual excess (MWh)', 'Annual excess per solar site (kWh)', 'log(1 + excess per site)']
            + [LABELS[v] for v in X_ALL])
desc['skew'] = d[desc.index].skew()
OUTC = ['uptake_pct', 'battery_kWh_per_solar_site', 'uptake_all_pct', 'battery_kWh_per_solar_site_all', 'new_uptake_pp',
        'need_p90_kWh_per_site', 'need_p75_kWh_per_site', 'need_median_kWh_per_site', 'pct_days_with_excess',
        'coverage_p90', 'coverage_p75', 'coverage_median',
        'above_ceg_kWh_per_site', 'pct_reverse_above_ceg']
o = d[OUTC].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
desc = pd.concat([desc, o.assign(description=OUTC)])
desc.round(3).to_csv(os.path.join(OUT, '2_descriptive_statistics.csv'))
pd.DataFrame({'variable': list(LABELS), 'description': list(LABELS.values()),
              'theme': [THEME.get(v, '') for v in LABELS]}).to_csv(os.path.join(OUT, 'variable_dictionary.csv'), index=False)
print('Network status:', d['network_status_detail'].value_counts().to_dict())
print(f"Uptake (4-quarter mean): residential median {d['uptake_pct'].median():.1f}%, all segments {d['uptake_all_pct'].median():.1f}%")
for lab in ['p90', 'p75', 'median']:
    c = d[f'coverage_{lab}'].dropna()
    print(f"Coverage {lab}: {len(c)} postcodes with need, median {c.median():.2f}, covered (>= 1) {(c >= 1).mean():.0%}, "
          f"total battery / total need {d.loc[c.index, 'battery_kWh_all'].sum() / d.loc[c.index, f'need_{lab}_kWh'].sum():.2f}")

print('Saved to', OUT)

### STEP 1 - Social variables per postal area (POA) from the ABS 2021 Census General Community Profile, Queensland.

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
OUT = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)


def read(table):
    df = pd.read_csv(os.path.join(DATA, f'2021Census_{table}_QLD_POA.csv'))
    df.columns = df.columns.str.strip()
    df['POSTCODE'] = df['POA_CODE_2021'].str.replace('POA', '', regex=False).str.zfill(4)
    return df.set_index('POSTCODE')


def pct(num, den):
    return 100 * num / den.where(den > 0)


g02, g04a, g04b, g06, g29 = read('G02'), read('G04A'), read('G04B'), read('G06'), read('G29')
g37, g43, g60 = read('G37'), read('G43'), read('G60B')

s = pd.DataFrame(index=g02.index)
inc = g02['Median_tot_hhd_inc_weekly'].where(g02['Median_tot_hhd_inc_weekly'] > 0)
s['median_hhd_income_weekly'] = inc

pop = g04b['Tot_P']
s['pct_age_20_29'] = pct(g04a[['Age_yr_20_24_P', 'Age_yr_25_29_P']].sum(axis=1), pop)
s['pct_over_65'] = pct(g04b[['Age_yr_65_69_P', 'Age_yr_70_74_P', 'Age_yr_75_79_P', 'Age_yr_80_84_P', 'Age_yr_85_89_P',
                             'Age_yr_90_94_P', 'Age_yr_95_99_P', 'Age_yr_100_yr_over_P']].sum(axis=1), pop)
s['pct_families_children'] = pct(g29['CF_ChU15_a_Total_F'] + g29['OPF_ChU15_a_Total_F'], g29['Total_F'])

dw = g37['Total_Total']
s['pct_owned_outright'] = pct(g37['O_OR_Total'], dw)
s['pct_owned_mortgage'] = pct(g37['O_MTG_Total'], dw)
s['pct_rented'] = pct(g37['R_Tot_Total'], dw)
s['pct_house'] = pct(g37['Total_DS_Sep_house'], dw)
s['pct_apartment'] = pct(g37['Total_DS_Flat_apart'], dw)

s['pct_married'] = pct(g06['P_Tot_Marrd_reg_marrge'] + g06['P_Tot_Married_de_facto'], g06['P_Tot_Total'])
p15 = g43['P_15_yrs_over_P']
s['pct_bachelor_or_higher'] = pct(g43['non_sch_qual_PostGrad_Dgre_P'] + g43['non_sch_qual_Gr_Dip_Gr_Crt_P']
                                  + g43['non_sch_qual_Bchelr_Degree_P'], p15)
s['pct_trades_workers'] = pct(g60['P_Tot_TechnicTrades_W'], g60['P_Tot_Tot'])
s['pct_fulltime_15plus'] = pct(g43['lfs_Emplyed_wrked_full_time_P'], p15)
s['pct_unemployed_15plus'] = pct(g43['lfs_Unmplyed_lookng_for_wrk_P'], p15)
s['pct_not_in_labour_force'] = pct(g43['lfs_N_the_labour_force_P'], p15)
s['population'] = pop
s['dwellings'] = dw

s.round(3).to_csv(os.path.join(OUT, 'social_variables.csv'))
print(f"{len(s)} Queensland postal areas, {s.shape[1]} columns")
print(s.describe().T[['count', 'mean', 'min', 'max']].round(2).to_string())

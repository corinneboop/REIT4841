### Shared list of the non-overlapping social variables (built in 1_social_variables; see its header for the
### definitions and for the variables deliberately not built because they overlap).
### Categories in a theme are compositional (they add up to <= 100%), so in a MULTIVARIABLE model leave one category
### of each such theme out (REFERENCE_OMIT).

NON_OVERLAPPING = {
    'Income': ['median_hhd_income_weekly'],
    'Age (persons)': ['pct_age_20_29', 'pct_over_65'],
    'Family (families)': ['pct_families_children'],
    'Tenure (dwellings)': ['pct_owned_outright', 'pct_owned_mortgage', 'pct_rented'],
    'Dwelling type (dwellings)': ['pct_house', 'pct_apartment'],
    'Relationship (persons 15+)': ['pct_married'],
    'Education (persons 15+)': ['pct_bachelor_or_higher'],
    'Occupation (employed)': ['pct_trades_workers'],
    'Employment (persons 15+)': ['pct_fulltime_15plus', 'pct_unemployed_15plus', 'pct_not_in_labour_force'],
}
SOCIAL = [v for vs in NON_OVERLAPPING.values() for v in vs]

REFERENCE_OMIT = ['pct_owned_mortgage']
MULTIVARIABLE_SET = [v for v in SOCIAL if v not in REFERENCE_OMIT]

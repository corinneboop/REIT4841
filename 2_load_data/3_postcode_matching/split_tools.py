### Shared by 3_mesh_block_split (3.3), 4_voronoi_split (3.4) and 5_compare_methods (3.5): the business label and how a
### site's half-hourly load is shared between its postcodes. Forward load (MW >= 0) follows the method's load SHARE;
### reverse flow (MW < 0) follows SOLAR_SHARE, the share of the site's rooftop solar in each postcode
### (2_network_matching/site_postcode_solar.csv).

import numpy as np
import pandas as pd

BUSINESS = ['Commercial', 'Industrial', 'Education', 'Hospital/Medical']
BUSINESS_TYPES = ['C&I', 'GROUND']
MIN_MISSING_FRACTION = 0.05


def solar_shares(m, solar_file):
    """SOLAR_SHARE(s, p): share of site s's rooftop solar that is in postcode p (2_network_matching), kept to the site-postcode pairs of this method and renormalised; a site with no solar behind it uses its load SHARE."""
    sp = pd.read_csv(solar_file, dtype={'POSTCODE': str}).set_index(['SITE', 'POSTCODE'])['SOLAR_KVA']
    kva = pd.Series([sp.get(k, 0.0) for k in zip(m['SITE'], m['POSTCODE'])], index=m.index)
    total = kva.groupby(m['SITE']).transform('sum')
    return (kva / total.replace(0, np.nan)).fillna(m['SHARE'])


def postcode_load(X, S, SS, PF, min_missing=MIN_MISSING_FRACTION):
    """Half-hourly postcode load: each site's forward load (MW >= 0) split by the load SHARE, its reverse flow (MW < 0) by SOLAR_SHARE."""
    ok = (X.isna().astype(float) @ PF) < min_missing
    fwd = X.clip(lower=0).fillna(0) @ S
    rev = (-X.clip(upper=0)).fillna(0) @ SS
    return (fwd - rev).where(ok), rev.where(ok)


def postcode_files(mapping, site_load):
    """Half-hourly postcode net load and reverse flow (MW) from a site -> postcode mapping (SHARE, SOLAR_SHARE, POSTCODE_FRACTION) and the cleaned site load."""
    def matrix(col):
        return mapping.pivot_table(index='SITE', columns='POSTCODE', values=col, aggfunc='sum').fillna(0)
    S = matrix('SHARE')
    SS, PF = matrix('SOLAR_SHARE').reindex_like(S).fillna(0), matrix('POSTCODE_FRACTION').reindex_like(S).fillna(0)
    return postcode_load(site_load[S.index], S, SS, PF)

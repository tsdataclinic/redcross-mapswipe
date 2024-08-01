import pandas as pd
from pysal.explore import esda
from pysal.lib import weights


def calc_moran_for_k(gdf_agg, col_name, k_vals=(1, 3, 5, 10, 15, 20, 25, 30)):
    moran_vals = {}
    for k in k_vals:
        w = weights.KNN.from_dataframe(gdf_agg, k=k)
        w.transform = "R"
        moran = esda.moran.Moran(gdf_agg[col_name], w)
        moran_vals[k] = moran
    return pd.DataFrame({"k": list(moran_vals.keys()), "moran_i": [m.I for m in moran_vals.values()]}), moran_vals

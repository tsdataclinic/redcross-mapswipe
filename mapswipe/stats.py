import pandas as pd
from pysal.explore import esda
from pysal.lib import weights

from mapswipe.data import get_project_data


def calc_moran_for_dist(gdf_agg, col_name, dist_vals):
    moran_vals = {}
    for dist in dist_vals:
        print(f"moran for {dist=}")
        w = weights.DistanceBand.from_dataframe(gdf_agg, threshold=dist)
        w.transform = "R"
        moran = esda.moran.Moran(gdf_agg[col_name], w)
        moran_vals[dist] = moran
    return pd.DataFrame({"dist": list(moran_vals.keys()), "moran_i": [m.I for m in moran_vals.values()]}), moran_vals


def calc_moran_for_knn(gdf_agg, col_name, k_vals=(1, 3, 5, 10, 15, 20, 25, 30)):
    moran_vals = {}
    for k in k_vals:
        w = weights.KNN.from_dataframe(gdf_agg, k=k)
        w.transform = "R"
        moran = esda.moran.Moran(gdf_agg[col_name], w)
        moran_vals[k] = moran
    return pd.DataFrame({"k": list(moran_vals.keys()), "moran_i": [m.I for m in moran_vals.values()]}), moran_vals


def safe_calc_moran(project_id, moran_func):
    data = get_project_data(project_id)
    if data.get("agg") is None:
        print(f"Skipping calculating moran for {project_id}")
        return None
    print(f"Calculating moran for {project_id}")
    try:
        df_moran_proj, moran_objs = moran_func(data["agg"])
    except:
        print(f"ERROR calculating moran for {project_id}")
        return None
    df_moran_proj["project_id"] = project_id
    return df_moran_proj

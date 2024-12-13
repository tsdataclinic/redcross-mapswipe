from importlib.resources import files, as_file
import math
import pandas as pd
import numpy as np
from pysal.explore import esda
from pysal.lib import weights
from statsmodels.formula.api import ols
from tqdm.notebook import tqdm

from mapswipe.data_access import read_raw_full_results, read_raw_agg_results, augment_agg_results


#
# Workflow configuration
# 

# Neighbor distance in meters for calculating Local Moran's I
MORAN_DISTANCE_METERS = 500.0

# Dependent variable for calculating spatial correlation
DEPENDENT_VAR = "adjusted_remap_score"

# Mapping difficulty model features
MODEL_FEATURES = [
    "geom_segment_count",
    "nearby_building_count_log",
    "building_area_m2",
    "aspect_ratio",
    "coverage_ratio",
]

# H3 resolution level for hex-aggregated Moran's I quadrant visualization
HEX_VIZ_H3_RESOLUTION = 11


# Acceptable threshold types and max value validation
THRESHOLD_TYPES = {
    "Top N Tasks": lambda df: len(df),
    "Top N% of Tasks": lambda _: 100,
    "Tasks With >=N remap_score": lambda _: 1.0,
}


# Remap selection column name
SELECTION_COL = "_selected"


# Internal logic

_OFFSET_RESPONSE = 3


def _get_user_metrics():
    with as_file(files("mapswipe.data").joinpath("user-metrics.csv")) as f:
        return pd.read_csv(f)


def _get_project_agg_weighted(df_agg, df_full, df_user_metrics):
    # Suffix the unweighted measures with _uw and write the weighted measures in their place
    df_agg = df_agg.copy()
    df_agg = df_agg.rename((
        {f"{i}_count": f"{i}_count_uw" for i in range(4)}
        | {f"{i}_share": f"{i}_share_uw" for i in range(4)}
        | {"total_count": "total_count_uw", "incorrect_score": "incorrect_score_uw"}
    ), axis=1)

    df_full_weight = df_full.merge(df_user_metrics[["user_id", "user_weight"]], on="user_id", how="left")
    df_full_weight["user_weight"] = df_full_weight["user_weight"].fillna(1.0)
    df_task_weight = df_full_weight.groupby(["project_id", "task_id", "result"]).agg({"user_weight": "sum"}).round()
    df_agg_weight_share = df_task_weight.reset_index().pivot_table(values="user_weight", index=["project_id", "task_id"], columns="result", fill_value=0.0)
    if _OFFSET_RESPONSE not in df_agg_weight_share.columns:
        df_agg_weight_share[_OFFSET_RESPONSE] = 0.0
    w_count_cols = {i: f"{i}_count" for i in df_agg_weight_share.columns}
    df_agg_weight_share = df_agg_weight_share.rename(columns=w_count_cols)
    df_agg_weight_share["total_count"] = df_agg_weight_share[w_count_cols.values()].sum(axis=1)
    for i in range(4):
        df_agg_weight_share[f"{i}_share"] = df_agg_weight_share[f"{i}_count"] / df_agg_weight_share["total_count"]
    df_agg = df_agg.set_index(["project_id", "task_id"]).join(df_agg_weight_share, how="left").reset_index()

    # Calculate the weighted and unweighted remap score for each row
    category_weights = {
        "0": 0.6,
        "2": 0.3,
        "3": 0.1,
    }
    for suffix in ("", "_uw"):
        df_agg[f"remap_score{suffix}"] = np.sum(
            (df_agg[[f"{c}_count{suffix}" for c in category_weights.keys()]] * np.array(list(category_weights.values()))),
            axis=1,
        ) / df_agg[f"total_count{suffix}"]
        df_agg[f"remap_score{suffix}"] = df_agg[f"remap_score{suffix}"] / df_agg[f"remap_score{suffix}"].max()

    return df_agg


def _moran_sig_quads(ser_tasks, lisa):
    sig = 1 * (lisa.p_sim < 0.05)
    spots = lisa.q * sig
    return pd.Series(spots, index=ser_tasks)


def _calc_moran_local_for_dist(gdf_agg, col_name, dist_vals):
    moran_vals = {}
    # Project to UTM for distance calculation
    task_ids = gdf_agg["task_id"]
    gdf = gdf_agg.to_crs(gdf_agg.estimate_utm_crs())
    for dist in dist_vals:
        w = weights.DistanceBand.from_dataframe(gdf, threshold=dist)
        w.transform = "R"
        moran = esda.moran.Moran_Local(gdf[col_name], w)
        moran_vals[f"moran_quad_{int(dist)}m"] = _moran_sig_quads(task_ids, moran)
    return pd.DataFrame(data=moran_vals, index=task_ids)





# Workflow orchestration

def _load_data(vars):
    project_id = vars["project_id"]
    df_agg_raw = read_raw_agg_results(project_id)
    df_agg_raw["project_id"] = project_id
    vars["df_agg_raw"] = df_agg_raw
    vars["df_full_raw"] = read_raw_full_results(project_id)
    return vars


def _compute_features(vars):
    vars["df_agg"] = augment_agg_results(vars["df_agg_raw"])
    return vars


def _apply_user_weighting(vars):
    vars["df_agg_w"] = _get_project_agg_weighted(
        vars["df_agg"],
        vars["df_full_raw"],
        _get_user_metrics(),
    )
    return vars


def _calc_adjusted_remap_score(vars):
    df_agg_w = vars["df_agg_w"]
    df_ols = df_agg_w.copy()
    # Center at the mean to improve interpretability of the intercept
    df_ols[MODEL_FEATURES] = df_ols[MODEL_FEATURES] - df_ols[MODEL_FEATURES].mean()
    model = ols("remap_score ~ " + " + ".join(MODEL_FEATURES), data=df_ols)
    results = model.fit()
    df_agg_w["adjusted_remap_score"] = results.resid
    
    vars["model"] = model
    vars["model_results"] = results
    return vars


def _calc_spatial_correlation(vars):
    df_agg_w = vars["df_agg_w"]
    df_moran_local_w = _calc_moran_local_for_dist(df_agg_w, DEPENDENT_VAR, [MORAN_DISTANCE_METERS])
    vars["df_agg_moran_w"] = df_agg_w.set_index("task_id").join(df_moran_local_w, how="inner").reset_index()
    return vars

def _calc_average_remap_score(vars):
    df_agg_moran_w = vars["df_agg_moran_w"]
    vars["df_avg_remap"] = pd.DataFrame(
        df_agg_moran_w[["remap_score_uw", "remap_score"]]
        .rename({"remap_score_uw": "Unweighted Remap Score", "remap_score": "User-Weighted Remap Score"}, axis=1)
        .mean()
    ).T.rename({0: "Average Value"})
    return vars


_WORKFLOW_STEPS = (
    (_load_data, "Loading project data"),
    (_compute_features, "Computing spatial features"),
    (_apply_user_weighting, "Applying user weighting to metrics"),
    (_calc_adjusted_remap_score, "Adjusting remap_score for mapping difficulty"),
    (_calc_spatial_correlation, "Calculating spatial correlation"),
    (_calc_average_remap_score, "Calculating average remap scores"),
)


def analyze_project(project_id):
    """Run the analysis workflow steps in a notebook context."""
    vars = {"project_id": project_id}
    with tqdm(total=len(_WORKFLOW_STEPS), desc="Analyzing Project", unit="step") as pbar:
        for func, step in _WORKFLOW_STEPS:
            pbar.set_description(step)
            vars = func(vars)
            pbar.update(1)
        pbar.set_description("Analysis complete")
    return vars


def _validate_threshold_args(gdf, threshold_n, threshold_type):
    if threshold_type not in THRESHOLD_TYPES:
        raise ValueError(f"Unsupported threshold type '{threshold_type}'")
    max_val = THRESHOLD_TYPES[threshold_type](gdf)
    if not (0 <= threshold_n <= max_val):
        raise ValueError(f"Threshold type '{threshold_type}' must be between 0 and {max_val}")


def apply_threshold_filter(gdf, threshold_n, threshold_type, selection_col):
    _validate_threshold_args(gdf, threshold_n, threshold_type)
    gdf = gdf.sort_values("remap_score").copy()
    if threshold_type == "Top N% of Tasks":
        threshold_n = int(len(gdf) * (threshold_n / 100.0))
    elif threshold_type == "Tasks With >=N remap_score":
        threshold_n = int(len(gdf[gdf["remap_score"] >= threshold_n]))
    gdf[selection_col] = 0.0
    gdf[selection_col].iloc[-threshold_n:] = 1.0
    return gdf


def generate_maproulette_shapefiles(gdf, project_id, file_size=10**6):
    task_size = len(gdf.set_index("task_id")["geometry"].head(10).to_json()) // 10
    chunk_size = file_size // task_size
    gdf = gdf.set_index("task_id")
    chunks = [gdf.iloc[i:i + chunk_size] for i in range(0, len(gdf), chunk_size)]
    files = []
    for idx, gdf_chunk in enumerate(chunks, start=1):
        file_name = f"remap_maproulette_{project_id}_file{idx}of{len(chunks)}.geojson"
        with open(file_name, "w", encoding="utf8") as f:
            f.write(gdf_chunk["geometry"].to_json())
        files.append(file_name)
    return files

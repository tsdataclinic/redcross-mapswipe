from importlib.resources import files, as_file
import pandas as pd


OFFSET_RESPONSE = 3


def get_user_metrics():
    with as_file(files("mapswipe.data").joinpath("user-metrics.csv")) as f:
        return pd.read_csv(f)


def get_project_agg_weighted(df_agg, df_full, df_user_metrics):
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
    if OFFSET_RESPONSE not in df_agg_weight_share.columns:
        df_agg_weight_share[OFFSET_RESPONSE] = 0.0
    w_count_cols = {i: f"{i}_count" for i in df_agg_weight_share.columns}
    df_agg_weight_share = df_agg_weight_share.rename(columns=w_count_cols)
    df_agg_weight_share["total_count"] = df_agg_weight_share[w_count_cols.values()].sum(axis=1)
    for i in range(4):
        df_agg_weight_share[f"{i}_share"] = df_agg_weight_share[f"{i}_count"] / df_agg_weight_share["total_count"]
    df_agg = df_agg.set_index(["project_id", "task_id"]).join(df_agg_weight_share, how="left").reset_index()

    # TODO improve this logic beyond the yes share
    df_agg["incorrect_score"] = 1 - df_agg["1_share"]

    return df_agg

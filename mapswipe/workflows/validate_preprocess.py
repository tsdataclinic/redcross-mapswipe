import multiprocessing as mp
import numpy as np
import pandas as pd

from mapswipe.data_access import read_scoped_projects_list, read_raw_full_results

# Github's ubuntu-latest only has 2 CPUs
DOWNLOAD_PARALLELISM = 2

def get_validate_population_data():
    """
    Load and calculate data for all validate projects. This function:
    1. Loads the list of MapSwipe projects
    2. Filters down to the in-scope validate projects
    3. Calculates user metrics (including weights) across in-scope validate projects

    Filtering criteria is available in mapswipe.data_access.read_scoped_projects_list

    :return: tuple of (DataFrame of project summaries, DataFrame of user metrics)
    """
    df_projects = read_scoped_projects_list()

    validate_projects = list(set(df_projects["project_id"]))
    with mp.Pool(processes=DOWNLOAD_PARALLELISM) as pool:
        data = pool.map(read_raw_full_results, validate_projects)
        all_proj_data = dict([(k, v) for k, v in zip(validate_projects, data)])
    df_full_all = pd.concat(all_proj_data.values())
    df_user_stats = df_full_all.drop_duplicates().groupby("user_id").agg(
        involved_project_count=("project_id", "nunique"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    ).reset_index()
    df_user_stats["first_seen"] = pd.to_datetime(df_user_stats["first_seen"], format="mixed").dt.floor("min")
    df_user_stats["last_seen"] = pd.to_datetime(df_user_stats["last_seen"], format="mixed").dt.floor("min")
    df_user_stats["tenure_days"] = (df_user_stats["last_seen"] - df_user_stats["first_seen"]).apply(lambda x: x.days + 1)
    df_user_stats["user_weight"] = df_user_stats["involved_project_count"].apply(lambda x: max(1, np.log2(x)))

    # Adjust for more stable outputs
    df_projects = df_projects.drop(["idx"], axis=1).sort_values("project_id")
    df_user_stats = df_user_stats.sort_values("user_id")

    return df_projects, df_user_stats


def main():
    print("Starting validate project data processing")
    root_path="mapswipe/data"
    df_projects, df_user_stats = get_validate_population_data()
    
    output_path = f"{root_path}/validate-project-summaries.csv"
    df_projects.to_csv(output_path, index=False)
    print(f"Wrote {len(df_projects)} project summaries to {output_path}")

    output_path = f"{root_path}/user-metrics.csv"
    df_user_stats.to_csv(output_path, index=False)
    print(f"Wrote {len(df_user_stats)} user metric records to {output_path}")


if __name__ == "__main__":
    main()

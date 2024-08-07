import diskcache
import io
import geopandas as gpd
import gzip
import pandas as pd
import requests

PROJECTS_DATA = "https://apps.mapswipe.org/api/projects/projects.csv"
PROJECTS_GEO = "https://apps.mapswipe.org/api/projects/projects_geom.geojson"

# Change for your situation
CACHE_PATH = "/Users/dsantin/Documents/Data Clinic/mapswipe-data"
CACHE_SIZE = 100 * 1e9

IGNORE_PROJECTS = (
    "-MRL3frZWPOCR94ehFnp", # seems like a synthetic project, https://download.geoservice.dlr.de/WSF2019/
    # These don't load for some reason
    "-NcESqSR6b0xA_FcPEwx",
    "-NcET8HgshJ837e4jS8r",
    "-NcETS2YIThvPut0CsZt",
    "-N0b1pvuEOrIrMfH6KnW",
    "-N6P-QAtJ7HO4Vwfr8OL",
    "-MxuPfkJp-w83wt2LT0v",
    "-NAtRt8B99CmTn9H0oKO",
)


def read_raw_projects_list():
    with io.BytesIO(requests.get(PROJECTS_DATA).content) as b:
        return pd.read_csv(b)


def read_scoped_projects_list():
    df = read_raw_projects_list()
    # validate projects only
    df = df[df["project_type"] == 2]
    # ignore outliers
    df = df[~df["project_id"].isin(IGNORE_PROJECTS)]
    # only finished
    df = df[df["status"] == "finished"]
    return df


def read_raw_full_results(project_code):
    url = f"https://apps.mapswipe.org/api/results/results_{project_code}.csv.gz"

    print(f"Downloading {url}")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Skipping full download for {project_code} - HTTP error {response.status_code}")
        return None
    gzipped_file = io.BytesIO(response.content)

    with gzip.GzipFile(fileobj=gzipped_file) as f:
        results = pd.read_csv(f)
    return results


def read_raw_agg_results(project_code):
    url = f"https://apps.mapswipe.org/api/agg_results/agg_results_{project_code}_geom.geojson.gz"
    print(f"Downloading {url}")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Skipping agg download for {project_code} - HTTP error {response.status_code}")
        return None
    gzipped_file = io.BytesIO(response.content)

    with gzip.GzipFile(fileobj=gzipped_file) as f:
        gdf = gpd.read_file(f)

    return gdf


def calc_agreement(row: pd.Series) -> float:
    """
    for each task the "agreement" is computed (i.e. the extent to which
    raters agree for the i-th subject). This measure is a component of
    Fleiss' kappa: https://en.wikipedia.org/wiki/Fleiss%27_kappa
    """

    n = row["total_count"]

    row = row.drop(labels=["total_count"])
    # extent to which raters agree for the ith subject
    # set agreement to None if only one user contributed
    if n == 1 or n == 0:
        agreement = None
    else:
        agreement = (sum([i**2 for i in row]) - n) / (n * (n - 1))

    return agreement


def calc_nearby_buildings(row: gpd.GeoSeries, all_centroids: gpd.GeoSeries, threshold_m: float) -> int:
    # row and all_centroids must already be centroids
    radius = row["centroid"].buffer(distance=threshold_m)
    others = all_centroids[radius.contains(all_centroids)]
    return len(others) - 1  # exclude itself


AGG_DEFAULTS = {
    "osm_username": "",
    "lastEdit": None,
}


def augment_agg_results(gdf):
    if gdf is None:
        return None
    replacement_dict = {0: "no", 1: "yes", 2: "unsure", 3: "offset"}
    for col, default_val in AGG_DEFAULTS.items():
        if col not in gdf.columns:
            gdf[col] = default_val

    gdf = gdf[['idx', 'task_id', '0_count', '1_count', '2_count', '3_count', '0_share', '1_share', '2_share', '3_share', 'total_count', "lastEdit", "osm_username", "geometry"]].copy()

    gdf["agreement"] = gdf[["total_count", '0_count', '1_count', '2_count', '3_count']].apply(calc_agreement, axis = 1)

    gdf["lastEdit"] = pd.to_datetime(gdf["lastEdit"])
    gdf["year"] = gdf["lastEdit"].dt.year
    gdf["modal_answer"] = gdf[['0_count', '1_count', '2_count', '3_count']].idxmax(axis=1)
    gdf["modal_answer"] = gdf["modal_answer"].replace(replacement_dict)
    gdf["yes_building"] = gdf["modal_answer"] == "1_count"

    # TODO improve this logic beyond the yes share
    gdf["incorrect_score"] = 1 - gdf["1_share"]

    # Calculate projected measures
    input_crs = gdf.crs  # should be 4327
    gdfp = gdf.to_crs(gdf.estimate_utm_crs())

    gdfp["centroid"] = gdfp.centroid
    gdfp["nearby_building_count"] = gdfp.apply(
        calc_nearby_buildings, 
        axis=1, 
        all_centroids=gdfp["centroid"],
        threshold_m=100.0,
    )
    gdfp = gdfp.drop("centroid", axis=1)

    gdfp["building_area_m2"] = gdfp.geometry.area
    
    gdf = gdfp.to_crs(input_crs)

    return gdf


def get_project_data(project_id):
    with diskcache.Cache(directory=CACHE_PATH, size_limit=CACHE_SIZE) as cache:
        if project_id not in cache:
            data = {
                "full": read_raw_full_results(project_id),
                "agg_raw": read_raw_agg_results(project_id),
            }
            cache[project_id] = data
        data = cache[project_id]
    data["agg"] = augment_agg_results(data["agg_raw"])
    del data["agg_raw"]
    return data

import diskcache
import io
import geopandas as gpd
import gzip
import numpy as np
import pandas as pd
import requests
from pysal.lib import weights

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


def calc_nearby_buildings(gdfp: gpd.GeoDataFrame, threshold_m: float) -> pd.Series:
    w = weights.DistanceBand.from_dataframe(
        gdfp.set_geometry("centroid"),
        threshold=threshold_m,
        binary=True,
        silence_warnings=True,
    )
    return w.full()[0].sum(axis=0).astype(int)


def count_polygon_segments(geometry):
    if geometry.geom_type == 'Polygon':
        exterior_segments = len(geometry.exterior.coords) - 1
        interior_segments = sum(len(interior.coords) - 1 for interior in geometry.interiors)
        return exterior_segments + interior_segments
    elif geometry.geom_type == 'MultiPolygon':
        return sum(count_polygon_segments(polygon) for polygon in geometry.geoms)
    else:
        return 0  # Not a polygon


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

    gdf = gdf[['idx', 'project_id', 'task_id', '0_count', '1_count', '2_count', '3_count', '0_share', '1_share', '2_share', '3_share', 'total_count', "lastEdit", "osm_username", "geometry"]].copy()

    gdf["agreement"] = gdf[["total_count", '0_count', '1_count', '2_count', '3_count']].apply(calc_agreement, axis = 1)

    gdf["lastEdit"] = pd.to_datetime(gdf["lastEdit"])
    gdf["year"] = gdf["lastEdit"].dt.year
    gdf["modal_answer"] = gdf[['0_count', '1_count', '2_count', '3_count']].idxmax(axis=1)
    gdf["modal_answer"] = gdf["modal_answer"].replace(replacement_dict)
    gdf["yes_building"] = gdf["modal_answer"] == "1_count"

    gdf["geom_segment_count"] = gdf["geometry"].apply(count_polygon_segments)

    # Calculate projected measures
    input_crs = gdf.crs  # should be 4327
    gdfp = gdf.to_crs(gdf.estimate_utm_crs())

    gdfp["centroid"] = gdfp.centroid
    gdfp["nearby_building_count"] = calc_nearby_buildings(gdfp, threshold_m=500.0)
    gdfp["nearby_building_count_log"] = gdfp["nearby_building_count"].apply(lambda x: max(0, np.log10(x)))
    gdfp = gdfp.drop("centroid", axis=1)

    gdfp["building_area_m2"] = gdfp.geometry.area

    b = gdfp.geometry.bounds
    dims = ["x", "y"]
    for dim in dims:
        b[dim] = b[f"max{dim}"] - b[f"min{dim}"]
    gdfp["aspect_ratio"] = b[dims].min(axis=1) / b[dims].max(axis=1)
    gdfp["box_area_m2"] = b["x"] * b["y"]
    gdfp["coverage_ratio"] = gdfp["building_area_m2"] / gdfp["box_area_m2"]
    
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

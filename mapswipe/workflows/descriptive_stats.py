import os
import geopandas as gpd
import gzip
from io import BytesIO
import requests
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import h3
import folium
import folium
from shapely.geometry import Polygon
import branca.colormap as cm
from folium.features import GeoJsonTooltip
from plotnine import options
from plotnine import theme_set, themes
from IPython.display import display, Markdown


replacement_dict = {0: "no", 1: "yes", 2: "unsure", 3: "offset"}

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

def read_agg_results(project_code):
    url = f"https://apps.mapswipe.org/api/agg_results/agg_results_{project_code}_geom.geojson.gz"
    response = requests.get(url)
    gzipped_file = BytesIO(response.content)

    with gzip.GzipFile(fileobj=gzipped_file) as f:
        gdf = gpd.read_file(f)

    gdf = gdf[['idx', 'task_id', '0_count', '1_count', '2_count', '3_count', '0_share', '1_share', '2_share', '3_share',
        'total_count', "lastEdit", "osm_username", "geometry"]]

    gdf["agreement"] = gdf[["total_count", '0_count', '1_count', '2_count', '3_count']].apply(calc_agreement, axis = 1)

    gdf["year"] = gdf["lastEdit"].dt.year
    gdf["modal_answer"] = gdf[['0_count', '1_count', '2_count', '3_count']].idxmax(axis=1)
    gdf["modal_answer"] = gdf["modal_answer"].replace(replacement_dict)
    gdf["yes_building"] = gdf["modal_answer"] == "1_count"

    return gdf

def read_full_results(project_code):
    url = f"https://apps.mapswipe.org/api/results/results_{project_code}.csv.gz"

    response = requests.get(url)
    gzipped_file = BytesIO(response.content)

    with gzip.GzipFile(fileobj=gzipped_file) as f:
        results = pd.read_csv(f)

    results["result"] = results["result"].replace(replacement_dict)

    return results

def create_hex_map(task_gdf, h3_resolution):
    gdf = task_gdf.copy(deep = True)
    gdf["geometry"] = gdf.centroid

    # Define hexagons
    def latlon_to_hexagon(row, resolution):
        return h3.geo_to_h3(row.geometry.y, row.geometry.x, resolution)

    gdf['hexagon'] = gdf.apply(latlon_to_hexagon, resolution=h3_resolution, axis=1)

    hexagon_gdf = gdf.groupby('hexagon').agg({"yes_building" : "mean", "task_id" : "nunique"}).reset_index()

    def hexagon_to_geometry(hexagon):
        vertices = h3.h3_to_geo_boundary(hexagon, geo_json=True)
        return Polygon(vertices)

    hexagon_gdf['geometry'] = hexagon_gdf['hexagon'].apply(hexagon_to_geometry)

    hexagon_gdf = gpd.GeoDataFrame(hexagon_gdf, geometry='geometry').set_crs(4326)

    # Create the map
    m = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=8)

    hexagon_geojson = hexagon_gdf.to_json()

    tooltip = GeoJsonTooltip(
        fields=['hexagon', 'task_id', 'yes_building'],
        aliases=['Hexagon ID:', 'Building Count:', "yes_pct"],  # These are the names that will appear in the tooltip
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )


    # Add Choropleth layer
    folium.Choropleth(
        geo_data=hexagon_geojson,
        name='choropleth',
        data=hexagon_gdf,
        columns=['hexagon', 'yes_building'],
        key_on='feature.properties.hexagon',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='% buildings positively identified'
    ).add_to(m)

    folium.GeoJson(
        hexagon_geojson,
        style_function=lambda x: {"fillColor": "YlOrRd", "color": "black", "weight": 1, "fillOpacity":0},
        tooltip=tooltip
    ).add_to(m)

    m._repr_html_ = lambda: m._parent._repr_html_(
    include_link=False, width='75%', height='400px'
    )
    return m

def create_task_map(gdf):

    geojson_data = gdf.drop('lastEdit', axis=1).to_json()

    map = folium.Map(location=[0.5, 0.5], zoom_start=8)
    map._repr_html_ = lambda: map._parent._repr_html_(
    include_link=False, width='75%', height='400px'
    )

    colormap = cm.linear.YlOrRd_09.scale(gdf["1_share"].min(), gdf["1_share"].max())

    def style_function(feature):
        return {
            'fillColor': colormap(feature['properties']['1_share']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.8
        }

    folium.GeoJson(
        geojson_data,
        style_function=style_function,
        name="geojson"
    ).add_to(map)

    colormap.add_to(map)

    return map


def make_outputs(project_code):
    out_dict = {}
    task_gdf = read_agg_results(project_code)

    total_tasks = task_gdf.shape[0]

    markdown_text = f"### Total number of tasks: {total_tasks}"
    out_dict["total_tasks"] = Markdown(markdown_text)

    p_counts = (ggplot(task_gdf, aes(x = "total_count")) + geom_histogram(binwidth = 1) + labs(x = "Number of responses", y = "Number of tasks"))
    out_dict["figure_task_counts"] = p_counts

    p_tasks_by_year = (ggplot(task_gdf, aes(x = "year")) + geom_histogram(binwidth = 1) + labs(x = "Year of addition to OSM", y = "Number of tasks"))
    out_dict["figure_task_years"] = p_tasks_by_year

    results = read_full_results(project_code)
    results = results.merge(task_gdf[["task_id", "year"]])

    p_response_types = (ggplot(task_gdf, aes(x = "modal_answer")) + geom_bar() + labs(x = "Most frequent task response", y = "Number of tasks") + scale_x_discrete(labels = {"0_count" : "No building", "1_count" : "Building", "2_count" : "Unsure", "3_count" : "Offset"}))
    out_dict["figure_response_types"] = p_response_types

    p_responses_year = (ggplot(results, aes(x = "year", fill = "result")) + geom_bar() + scale_fill_discrete(name = "Response") + labs(x = "Year of addition to OSM", y = "Number of responses"))
    out_dict["figure_responses_year"] = p_responses_year
    return out_dict
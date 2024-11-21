import folium
import h3
from shapely.geometry import Polygon
import geopandas as gpd
from folium.features import GeoJsonTooltip
from typing import Iterable
import branca.colormap as cm
from branca.element import MacroElement
from jinja2 import Template


# LISA colors
# https://geographicdata.science/book/notebooks/07_local_autocorrelation.html
_LISA_COLORMAP = {
    "ns": "#5c5c5c", # Values of 0
    "HH": "#d7191c",  # Values of 1
    "LH": "#abd9e9",  # Values of 2
    "LL": "#2c7bb6",  # Values of 3
    "HL": "#fdae61",  # Values of 4
}
_LISA_COLORS = [
    _LISA_COLORMAP["ns"], 
    _LISA_COLORMAP["HH"], 
    _LISA_COLORMAP["LH"], 
    _LISA_COLORMAP["LL"], 
    _LISA_COLORMAP["HL"],
]


# Set to "Esri.WorldImagery" for satellite imagery
_MAP_TILE_PROVIDER = "OpenStreetMap"

STARTING_ZOOM_LEVEL = 9


class Legend(MacroElement):
    def __init__(self, color_map, title="Legend"):
        super().__init__()
        self._template = Template('''
        {% macro html(this, kwargs) %}
        <div style="
            position: fixed; 
            bottom: 50px; 
            right: 50px; 
            width: 120px; 
            height: auto; 
            background-color: white;
            border: 2px solid grey; 
            z-index:9999; 
            font-size:14px;
            ">
            <p style="text-align: center;"><strong>{{this.title}}</strong></p>
            {% for category, color in this.color_map.items() %}
            <p>
                <i class="fa fa-square" style="color:{{color}}"></i> {{category}}
            </p>
            {% endfor %}
        </div>
        {% endmacro %}
        ''')
        self.color_map = color_map
        self.title = title



def create_moran_quad_map(gdf_agg, color_col, value_cols, center_pt=None, include_legend=True):
    gdf = gdf_agg.copy()

    geojson_data = gdf.drop('lastEdit', axis=1).to_json()

    if center_pt is None:
        center_pt = gdf.to_crs(gdf.estimate_utm_crs()).dissolve().centroid.to_crs(4326)

    map = folium.Map(tiles=_MAP_TILE_PROVIDER, location=[center_pt.y, center_pt.x], zoom_start=STARTING_ZOOM_LEVEL)
    map._repr_html_ = lambda: map._parent._repr_html_(
        include_link=False, width='75%', height='400px'
    )

    tooltip = GeoJsonTooltip(
        fields=[color_col] + [v for v in value_cols],
        aliases=[color_col] + [f"{v} Value" for v in value_cols],
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
    
    def style_function(feature):
        fillval = feature['properties'][color_col]
        fillval = int(fillval)
        return {
            'fillColor': _LISA_COLORS[fillval],
            'color': 'black',
            'weight': 0.25,
            'fillOpacity': 0.8
        }

    folium.GeoJson(
        geojson_data,
        style_function=style_function,
        tooltip=tooltip,
        name="geojson"
    ).add_to(map)

    if include_legend:
        map.get_root().add_child(Legend(dict(enumerate(_LISA_COLORS)), "Local Moran Quadrant"))

    return map


def create_moran_quad_hex_map(gdf_agg, mode_col, value_cols, h3_resolution, include_legend=True):
    gdf = gdf_agg.copy(deep = True)
    gdf["geometry"] = gdf.centroid

    # Define hexagons
    def latlon_to_hexagon(row, resolution):
        return h3.geo_to_h3shape(row.geometry.y, row.geometry.x, resolution)

    gdf['hexagon'] = gdf.apply(latlon_to_hexagon, resolution=h3_resolution, axis=1)

    def _mode(s):
        m = s.mode()
        if isinstance(m, Iterable):
            m = m[0]
        return m
    
    hexagon_gdf = gdf.groupby('hexagon').agg(
        {mode_col : _mode, "task_id" : "nunique", "nearby_building_count": "mean"}
        | {v: "median" for v in value_cols}
    ).reset_index()
    hexagon_gdf[mode_col] = hexagon_gdf[mode_col].astype(int)

    def hexagon_to_geometry(hexagon):
        vertices = h3.h3_to_geo_boundary(hexagon, geo_json=True)
        return Polygon(vertices)

    hexagon_gdf['geometry'] = hexagon_gdf['hexagon'].apply(hexagon_to_geometry)
    hexagon_gdf = gpd.GeoDataFrame(hexagon_gdf, geometry='geometry').set_crs(4326)
    hexagon_geojson = hexagon_gdf.to_json()

    # Create the map
    m = folium.Map(tiles=_MAP_TILE_PROVIDER, location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=STARTING_ZOOM_LEVEL)

    tooltip = GeoJsonTooltip(
        fields=['hexagon', 'task_id', mode_col, 'nearby_building_count'] + [v for v in value_cols],
        aliases=['Hexagon ID', 'Hex Building Count', mode_col, "Avg Nearby Building Count"] + [f"Median {v} Value" for v in value_cols],
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

    def style_function(feature):
        fillval = feature['properties'][mode_col]
        fillval = int(fillval)
        return {
            'fillColor': _LISA_COLORS[fillval],
            'color': 'black',
            'weight': 0.25,
            'lineOpacity': 0.2,
            'fillOpacity': 0.7,
        }
    
    folium.GeoJson(
        hexagon_geojson,
        style_function=style_function,
        tooltip=tooltip
    ).add_to(m)

    if include_legend:
        m.get_root().add_child(Legend(dict(enumerate(_LISA_COLORS)), "Local Moran Quadrant"))

    m._repr_html_ = lambda: m._parent._repr_html_(
        include_link=False, width='75%', height='400px'
    )
    
    return m


def create_task_map(gdf_agg, color_col, value_cols, col_descs, selection_col=None, center_pt=None, color_bounds=None):
    gdf = gdf_agg.copy()

    geojson_data = gdf.drop('lastEdit', axis=1).to_json()

    if center_pt is None:
        center_pt = gdf.to_crs(gdf.estimate_utm_crs()).dissolve().centroid.to_crs(4326)
    if color_bounds is None:
        color_bounds = (gdf[color_col].min(), gdf[color_col].max())

    map = folium.Map(tiles=_MAP_TILE_PROVIDER, location=[center_pt.y, center_pt.x], zoom_start=STARTING_ZOOM_LEVEL)
    map._repr_html_ = lambda: map._parent._repr_html_(
        include_link=False, width='75%', height='400px'
    )

    if color_col not in value_cols:
        value_cols = list(value_cols) + [color_col]

    def _get_desc(col_name):
        if col_name in col_descs:
            return col_descs[col_name]
        return f"{col_name} Value"

    tooltip = GeoJsonTooltip(
        fields=[v for v in value_cols],
        aliases=[_get_desc(c) for c in value_cols],
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

    colormap = cm.linear.YlOrRd_09.scale(color_bounds[0], color_bounds[1])

    def style_function(feature):
        fill_opacity = 0.7
        line_opacity = 0.2
        if selection_col and not feature["properties"][selection_col]:
            fill_opacity = 0.01
            line_opacity = 0.01
        return {
            "fillColor": colormap(feature["properties"][color_col]),
            "color": "black",
            "weight": 0.25,
            "fillOpacity": fill_opacity,
            "lineOpacity": line_opacity,
        }

    folium.GeoJson(
        geojson_data,
        columns=[color_col],
        style_function=style_function,
        tooltip=tooltip,
        name="geojson"
    ).add_to(map)

    return map
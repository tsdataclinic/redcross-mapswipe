from setuptools import find_packages, setup

setup(
    name='mapswipe',
    packages=find_packages(),
    version='0.1.0',
    description='Library of functions for mapswipe validate project analytics',
    author='Data Clinic',
    install_requires=[
        "pysal",
        "pandas",
        "h3",
        "geopandas",
        "contextily",
        "diskcache",
        "folium",
        "branca",
        "fiona>=1.10b2", # macos pcre2 import error
        "tqdm",
        "ipywidgets",
    ],
    package_data={
        "mapswipe.data": ["*.csv"],
    },
)
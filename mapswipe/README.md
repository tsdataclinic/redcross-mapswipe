# MapSwipe Analysis

A Python package for analyzing MapSwipe building validation projects, with tools for spatial analysis and visualization.

## Project Structure

```
mapswipe/
├── data/
│   ├── __init__.py
│   ├── README.md
│   ├── user-metrics.csv
│   └── validate-project-summaries.csv
├── workflows/
│   ├── __init__.py
│   ├── descriptive_stats.py
│   ├── project_remap.py
│   ├── validate_preprocess.py
│   └── viz.py
├── __init__.py
├── data_access.py
├── project_stats.py
└── utils.py
```

## Core Components

### Workflow Modules
- `workflows/descriptive_stats.py`: Module for generating descriptive statistics and visualizations of MapSwipe validation projects.
- `workflows/project_remap.py`: Core analysis pipeline for processing MapSwipe building validation projects.
- `workflows/validate_preprocess.py`: Data preprocessing for validation projects. Script for generating population-level statistics about MapSwipe validate projects and users.
- `workflows/viz.py`: Visualization tools and mapping functions. Core visualization module for MapSwipe building validation data, providing interactive maps and statistical plots.

### Core Utilities
- `data_access.py`: Core module for interacting with MapSwipe's API and processing building validation data.
- `project_stats.py`: Project-level statistical calculations. Implements spatial autocorrelation analysis for MapSwipe building validation data using Moran's I statistics.



# Basic Usage Examples

## Workflow Modules

### descriptive_stats.py
```python
from mapswipe.workflows import descriptive_stats

# Generate standard project visualizations and metrics
project_code = "your-project-id"
outputs = descriptive_stats.make_outputs(project_code)

# Access specific outputs
total_tasks = outputs["total_tasks"]
response_plot = outputs["figure_response_types"]
yearly_plot = outputs["figure_responses_year"]
```

### project_remap.py
```python
from mapswipe.workflows import project_remap

# Run full analysis pipeline
project_id = "your-project-id"
results = project_remap.analyze_project(project_id)

# Access analysis results
weighted_data = results["df_agg_w"]
model_results = results["model_results"]
spatial_stats = results["df_agg_moran_w"]
```

### validate_preprocess.py
```python
from mapswipe.workflows import validate_preprocess

# Generate population-level statistics
df_projects, df_user_stats = validate_preprocess.get_validate_population_data()

# Write outputs to csv files
root_path = "mapswipe/data"
validate_preprocess.main()  # Writes to data directory
```

### viz.py
```python
from mapswipe.workflows import viz

# Create hexbin map
hex_map = viz.create_hex_map(task_gdf, h3_resolution=9)

# Create task-level map
task_map = viz.create_task_map(gdf_agg)

# Display maps in notebook
display(hex_map)
display(task_map)
```

## Core Utilities

### data_access.py
```python
from mapswipe import data_access

# Get list of projects
projects = data_access.read_scoped_projects_list()

# Load project data
project_data = data_access.get_project_data("your-project-id")

# Access components
raw_responses = project_data["full"]
aggregated_data = project_data["agg"]
```

### project_stats.py
```python
from mapswipe import project_stats

# Calculate Moran's I for specific distance
moran_results = project_stats.calc_moran_for_dist(
    gdf_agg,
    col_name="agreement",
    dist_vals=[500, 1000]  # meters
)

# K-nearest neighbors analysis
knn_results = project_stats.calc_moran_for_knn(
    gdf_agg,
    col_name="agreement",
    k_vals=(1, 3, 5, 10)
)
```

# Workflow Example

```python
# Complete analysis workflow
from mapswipe import data_access
from mapswipe.workflows import project_remap, viz

# 1. Get project data
project_id = "your-project-id"
project_data = data_access.get_project_data(project_id)

# 2. Run analysis
analysis_results = project_remap.analyze_project(project_id)

# 3. Create visualizations
hex_map = viz.create_hex_map(analysis_results["df_agg_moran_w"], h3_resolution=9)
display(hex_map)

# 4. Generate descriptive stats
from mapswipe.workflows import descriptive_stats
stats = descriptive_stats.make_outputs(project_id)
```
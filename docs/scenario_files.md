# Scenario Directory Structure

Reference for files produced at each pipeline stage within a scenario directory.

## Setup Inputs (`input_data/`)

- `analysis_areas.gpkg` — Study area boundary polygons with geometry IDs
- `osm_study_area.pbf` — OSM road network extract for the study area
- `user_classes.csv` — User class definitions (income bracket, car ownership, LTS tolerance, toll exemptions)
- `userclass_statistics.csv` — Population count per geometry × user class
- `GTFS/` — Transit feed archives
- `traffic/routing_graph.graphml` — OSM graph with traffic speed attributes (from TomTom conflation)
- `traffic/routing_edges.gpkg` — Same graph edges as GeoPackage for GIS inspection
- `census/census_tracts_with_dests.gpkg` — Tract geometries with destination counts (jobs, POIs)

## Representation Outputs (`routing/`)

- `user_classes_with_routeenvs.csv` — Mapping of each user class to its routing environment per mode
- `{routeenv}/osm_file.pbf` — Modified OSM network for this routing environment
- `{routeenv}/traffic/routing_graph.graphml` — Modified car graph (e.g., with income-specific toll penalties)
- `{routeenv}/r5_params.json` — R5 routing parameters (LTS threshold, walk speed)
- `{routeenv}/gtfs_files/` — GTFS files (copied from input, if transit is available in this environment)

## Routing Outputs

- `routing/{routeenv}/raw_ttms/raw_ttm_{MODE}.csv` — Raw travel time matrix (origins × destinations, minutes)
- `impedances/{userclass}/ttm_{MODE}.csv` — Post-processed travel time matrix for a user class
- `impedances/{userclass}/tcm_{MODE}.csv` — Travel cost matrix (dollars converted to minutes via value-of-time)
- `impedances/{userclass}/gtm_{MODE}.csv` — Generalized cost matrix (ttm + tcm combined)

## Evaluation Outputs (`results/`)

- `userclass_results.csv` — Aggregate accessibility metrics per user class (value sums, mode shares)
- `geometry_results.gpkg` — Per-geography results with geometries for mapping
- `geometry_results.html` — Interactive choropleth map with metric selector
- `geo_viz_by_userclass/` — Per-user-class result maps (.html + .gpkg)
- `detailed_data/lowest_traveltimes_by_userclass/` — Best travel time to each destination per user class
- `detailed_data/mode_selections_by_userclass/` — Which mode was chosen per O/D pair
- `detailed_data/values_per_dest_by_userclass/` — Accessibility value per destination per user class
- `detailed_data/value_sum_per_person_by_userclass/` — Per-person value aggregated by origin
- `detailed_data/value_sum_total_by_OD_by_userclass/` — Total value (value × population) per O/D pair
- `traffic/routing_graph_with_relative_demand.graphml` — Graph with estimated link volumes (if `track_volumes=True`)

## Comparison Outputs (`comparison/`)

When comparing two scenarios:

- `userclass_comparison.csv` — Side-by-side user class metrics
- `geometry_comparison.gpkg` — Per-geography difference values
- `value_comparison_map.html` — Map of accessibility value changes
- `mode_share_comparison_map.html` — Map of mode share changes
- `comparison_summary.txt` — Text summary of differences

## Data Formats

All matrices (TTM, TCM, GTM) are CSV with string geom_id row/column headers and values in **minutes** (costs converted via value-of-time). Population data uses `geom_id × user_class_id` structure. Geographic data uses GeoPackage (`.gpkg`).

## Key External Dependencies

| Dependency | Role |
|---|---|
| r5py | Transit, walk, bicycle routing (requires Java 21) |
| osmnx/networkx | OSM graph loading and car routing (Dijkstra) |
| census/pygris | Census demographics, boundaries, LODES employment |
| overturemaps | POI destination data |
| geopandas | Geospatial data throughout |
| folium | Interactive map visualization |

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Connectome measures general access-to-destinations in urban areas across all modes (car, transit, walk, bicycle, ridehail), all places, and all people. It targets USA cities at census tract level. The project is a pre-alpha refactor — actively developed, no formal package distribution yet.

This repo serves both as the development codebase and as the tool used to run analyses for clients. There is no separation between the two — client work frequently drives code enhancements.

## Environment

Conda environment is `cxome`. Java 21 must be available at runtime for r5py (multi-modal routing engine). Full install instructions are in README.md. No formal test suite — testing is done by running scenarios through `scenario_management.py` functions.

## Running the Code

All execution flows through `connectome/scenario_management.py`:

```python
from connectome.scenario_management import initialize_existing_conditions, run_scenario, compare_scenarios

# Set up a new study area
initialize_existing_conditions(scenario_dir, lat, lon, buffer_km, ...)

# Run a full analysis
run_scenario(scenario_dir, track_volumes=False)

# Compare two scenarios
compare_scenarios(study_dir, scenario1_name, scenario2_name, ...)
```

Test scenarios exist in `connectome/testing/` (denver40, denver20, lewes, burlington_test, pgh_test, pvd_test, ri_test) with pre-computed intermediate results.

## Architecture

### Pipeline Stages

The system runs as a linear pipeline, with each stage producing cached files that allow re-running from intermediate points:

1. **Setup** (`setup/`) — Downloads census boundaries, population demographics, employment (LODES), POIs (Overture), and road/transit networks (OSM, GTFS). Creates user classes from income × car-ownership matrix.

2. **Representation** (`representation.py` → `setup/define_experiences.py`) — Translates the objective physical network into subjective routing environments that reflect how different user classes experience the same infrastructure (e.g., tolls cost more minutes to lower-income drivers, bike networks differ by stress tolerance). Creates routing environment directories with modified network files, and returns a dataframe mapping each user class to its routing environment per mode. Multiple user classes often share the same environment for a given mode.

3. **Routing** (`routing_and_impedance.py` + `car_routing.py`) — Computes origin-destination travel time matrices. Car routing uses NetworkX Dijkstra on an OSM-derived graph with traffic speeds. Transit/walk/bike routing uses r5py. Post-processing adds mode-specific costs (parking, transit fares, fuel, tolls) and converts to generalized impedance matrices (time + money via value-of-time).

4. **Evaluation** (`evaluation.py`) — For each user class and O/D pair: picks the minimum-impedance mode, applies exponential decay (`exp(-0.05 * minutes)`), multiplies by destination counts, and aggregates results by geography and user class.

5. **Communication** (`communication.py`) — Generates user-facing data outputs, including geospatial and spreadsheet files and interactive Folium choropleth maps with radio-button metric selectors, plus comparison views for scenario analysis.

### Key Modules

- **`scenario_management.py`** — Top-level orchestrator. Entry point for all analyses. Manages directory structure and calls stage functions.
- **`routing_and_impedance.py`** — Routes all modes, applies impedance calculations (BPR volume-delay, tolls, parking, operating costs). Produces raw TTMs and per-userclass generalized cost matrices (GTMs).
- **`car_routing.py`** — NetworkX-based car routing with Dijkstra (scipy sparse or NetworkX fallback). Supports checkpoint tracking for induced demand analysis.
- **`evaluation.py`** — Mode choice, value calculation, geographic and demographic aggregation. Outputs `userclass_results.csv` and `geometry_results.gpkg`.
- **`communication.py`** — All visualization: choropleth maps, route visualizations, scenario comparisons. Uses Folium/branca for interactive HTML output.
- **`representation.py`** + **`setup/define_experiences.py`** — Creates routing environments from the physical network and maps user classes to them per mode. Outputs modified network files per environment and `user_classes_with_routeenvs.csv`.

### Setup Submodules

- **`setup/physical_conditions.py`** — OSM download/filter, LTS tagging for bikes, toll handling, GTFS processing.
- **`setup/census_geometry_usa.py`** — Census tract boundaries via Census API + Tigris.
- **`setup/populate_people_usa.py`** — ACS demographic data → user classes (income quintiles × car ownership).
- **`setup/populate_destinations.py`** — LODES jobs + Overture POIs aggregated to analysis geometries.
- **`setup/define_experiences.py`** — Maps user classes to routing environments and mode availability.

### Traffic Utilities (`traffic_utils/`)

- **`speed_utils.py`** — Conflates TomTom traffic speed data onto OSM network edges via name/bearing matching.
- **`volume_utils.py`** — Loads TMAS traffic monitoring station volumes and infers volumes on unmonitored links.
- **`assignment.py`** — Link-level traffic assignment, BPR volume-delay formula, induced demand calculation.
- **`traffic_analysis_parameters.json`** — Central config for functional class definitions, BPR alpha/beta, lane capacities, toll costs, parking costs, value-of-time multiplier (0.25 of wage).

For detailed file listings per scenario stage, see `docs/scenario_files.md`.

## Conventions

- No formal style guide yet. Match the style of surrounding code.
- All matrix values are in **minutes** — monetary costs are converted via value-of-time before storage.
- All geometry IDs (`geom_id`) are **strings**, not integers — cast carefully when merging.
- `traffic_analysis_parameters.json` is the central config for BPR, lane capacities, toll/parking costs, and value-of-time. Changes there affect routing and evaluation outputs.

## Gotchas

- r5py fails silently or throws opaque Java errors if not running Java 21.
- r5py sometimes throws errors if the r5 cache hasn't been cleared 
- Car routing graphs use `(u, v, 0)` edge keys (MultiDiGraph) — watch for key mismatches.
- Cached intermediate files mean re-running a stage won't pick up upstream changes unless you delete the cached outputs. When in doubt, delete the relevant output files and re-run.
- OSM PBF files are rewritten per routing environment — don't assume `osm_file.pbf` in a routeenv directory matches the original study area extract.

## Sensitive Files

`mapbox_token.txt` and `mobility_db_refresh_token.txt` in the connectome directory contain API credentials — do not commit or expose these.

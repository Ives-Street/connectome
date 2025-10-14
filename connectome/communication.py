import os
import pandas as pd
import geopandas as gpd
import folium
import json
import numpy as np
from branca.colormap import linear
from folium.features import GeoJson, GeoJsonTooltip
from folium.elements import MacroElement
from jinja2 import Template
import folium.plugins

MODES = [ #todo - make this universal for the whole codebase
    "CAR",
    "TRANSIT",
    "WALK",
    "BICYCLE",
    "RIDEHAIL",
]

def visualize_access_to_zone(scenario_dir,
                             geoms_with_dests,
                             outpath,
                             ttm,
                             target_zone_id = None,
                             ):
    """Visualize access to a specific zone or to the zone with most destinations.
    The target zone will be highlighted in red on the map.

    Args:
        geoms_with_dests: GeoDataFrame containing geometries and their destinations
        ttm: Travel time matrix (wide, with from_id index and to_id columns) or a CSV path to it
        target_zone_id: Target zone ID. If None, selects zone with most destinations
    """
    if target_zone_id is None:
        #   right now only using lodes_jobs
        total_dests = geoms_with_dests['lodes_jobs']
        # Get the zone with maximum destinations
        target_zone_id = total_dests.idxmax()

    # Load wide TTM if a path is provided
    if isinstance(ttm, str):
        ttm = pd.read_csv(ttm, index_col=0)

    # Ensure ids are strings for alignment with TTM
    target_zone_id = str(target_zone_id)
    # Some GeoDataFrames have geom_id as index; ensure it's also a column
    if "geom_id" not in geoms_with_dests.columns:
        geoms_with_dests = geoms_with_dests.copy()
        geoms_with_dests["geom_id"] = geoms_with_dests.index.astype(str)
    else:
        geoms_with_dests = geoms_with_dests.copy()
        geoms_with_dests["geom_id"] = geoms_with_dests["geom_id"].astype(str)

    # Extract travel times to target zone (a Series indexed by from_id/geom_id)
    if target_zone_id not in ttm.columns:
        raise KeyError(f"Target zone id {target_zone_id} not found in travel time matrix columns.")
    travel_times_to_dest = ttm[target_zone_id].copy()
    travel_times_to_dest.index.name = 'geom_id'
    travel_times_to_dest.name = 'travel_time_to_dest'

    # Merge travel times into the GeoDataFrame
    to_visualize = geoms_with_dests.merge(
        travel_times_to_dest,
        left_on="geom_id",
        right_index=True,
        how="left"
    )

    # Keep only the necessary columns
    to_visualize = to_visualize[['geom_id', 'geometry', 'travel_time_to_dest']]

    make_radio_choropleth_map(
        scenario_dir=scenario_dir,
        in_data=to_visualize,
        outfile=outpath,
        tiles="CartoDB.Positron",
        tooltip_precision=0,
    )

    print(
        f"Saved visualization of access to zone {target_zone_id} to {outpath}"
    )


def _coerce_and_prettify_numeric_columns(gdf: gpd.GeoDataFrame,
                                         exclude_cols: list = None) -> tuple[gpd.GeoDataFrame, dict]:
    """Coerce columns to numeric and rescale/rename for better display.

    Rescales numeric columns to reasonable ranges and updates column names to reflect
    the scale change. For example, values from 0-200,000 become 0-200 with column
    renamed to include "(k)" suffix.

    Args:
        gdf: GeoDataFrame to process
        exclude_cols: List of column names to exclude from processing

    Returns:
        tuple: (processed_gdf, column_mapping) where column_mapping maps
               original names to new names
    """
    gdf = gdf.copy()

    if exclude_cols is None:
        exclude_cols = []

    column_mapping = {}  # old_name -> new_name

    for col in gdf.columns:
        # Skip geometry and excluded columns
        if col == gdf.geometry.name or col in exclude_cols:
            continue

        # Try to coerce to numeric
        coerced = pd.to_numeric(gdf[col], errors="coerce")

        # If no numeric values, skip this column
        if not coerced.notna().any():
            continue

        gdf[col] = coerced.astype(float)

        # Determine appropriate scale and suffix
        non_null_values = gdf[col].dropna()

        if len(non_null_values) == 0:
            column_mapping[col] = col
            continue

        max_abs_value = non_null_values.abs().max()

        # Determine scale factor and suffix
        if max_abs_value >= 10_000_000:
            # Millions
            scale_factor = 1_000_000
            suffix = " (M)"
            decimals = 0
        elif max_abs_value >= 10_000:
            # Thousands
            scale_factor = 1_000
            suffix = " (k)"
            decimals = 0
        elif max_abs_value >= 1:
            # Keep as-is, but round to reasonable precision
            if max_abs_value > 100:
                scale_factor = 1
                suffix = ""
                decimals = 1
            else:
                scale_factor = 1
                suffix = ""
                decimals = 2
        else:
            # Small numbers (< 1), keep more precision
            scale_factor = 1
            suffix = ""
            decimals = 2

        # Apply scaling
        if scale_factor > 1:
            gdf[col] = gdf[col] / scale_factor

        # Round to appropriate decimal places
        if decimals > 0:
            gdf[col] = gdf[col].round(decimals)
        elif decimals == 0:
            # Convert to int, handling NaN/inf values
            gdf[col] = gdf[col].fillna(-999999).replace([np.inf, -np.inf], -999999)
            gdf[col] = gdf[col].round(0).astype(int)
            gdf[col] = gdf[col].replace(-999999, np.nan)

        # Create pretty column name
        # Convert snake_case to Title Case
        pretty_name = col.replace('_', ' ').title()
        new_name = f"{pretty_name}{suffix}"

        # Rename the column
        gdf = gdf.rename(columns={col: new_name})

        # Store mapping
        column_mapping[col] = new_name

    return gdf, column_mapping

#
def make_radio_choropleth_map(
        scenario_dir: str,
        in_data: gpd.GeoDataFrame | str,
        outfile: str | None = None,
        columns_to_viz: list[str] | None = None,
        initial: str | None = None,
        tiles: str = "CartoDB positron",
        tooltip_precision: int | None = None,
        exclude: list = [],
) -> folium.Map:
    """
    Build a Folium map where the user can choose exactly one numeric column
    (via radio buttons) to color the layer. The basemap never changes, and
    exactly one legend is shown at any time. Includes a north arrow and scale bar.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input geometries with attributes.
    numeric_columns : list[str] | None
        Subset of columns to expose. If None, all numeric columns are used (after coercion).
    initial : str | None
        Column selected on load. Defaults to the first numeric column.
    tiles : str
        Folium basemap.
    tooltip_precision : int | None
        If set, round numeric values in tooltip to this many decimals.
    exclude : list
        List of column names to exclude from processing (default: []).
    """
    # Load GeoDataFrame if file path provided
    if isinstance(in_data, str):
        gdf = gpd.read_file(os.path.join(scenario_dir, in_data))
    else:
        gdf = in_data

    if gdf.empty:
        raise ValueError("gdf is empty.")
    if gdf.geometry.name is None:
        raise ValueError("gdf has no active geometry column.")
    if not gdf.crs:
        # Folium expects WGS84; warn but continue
        pass

    # 1) Coerce all possible columns to float
    gdf, column_mapping = _coerce_and_prettify_numeric_columns(gdf, exclude_cols=exclude)

    if columns_to_viz is not None:
        columns_to_viz =[column_mapping[x] for x in columns_to_viz]

    # 2) Determine numeric columns
    auto_numeric = [c for c in gdf.columns if c != gdf.geometry.name and pd.api.types.is_numeric_dtype(gdf[c])]
    if columns_to_viz is None or columns_to_viz == []:
        numeric_columns = auto_numeric
    else:
        # keep only those that are actually numeric after coercion
        numeric_columns = [c for c in columns_to_viz if c in auto_numeric]

    if not numeric_columns:
        print("No numeric columns available to visualize after coercion. Defaulting to all columns.")
        #TODO fix error with determining these columns AFTER renaming columns above
        numeric_columns = auto_numeric

    # 3) Build the base map (basemap never changes)
    # Center on the data bounds
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    cx = (bounds[1] + bounds[3]) / 2
    cy = (bounds[0] + bounds[2]) / 2
    m = folium.Map(location=[cx, cy], zoom_start=12, tiles=tiles)

    # Add scale bar with double size
    scale = folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='meters',
                                          secondary_length_unit='miles')
    scale.add_to(m)

    # Add north arrow
    north_arrow = MacroElement()
    north_arrow._name = "north_arrow"

    # Create the HTML/CSS for the north arrow
    north_arrow.template = Template("""
        {% macro script(this, kwargs) %}
        var north_arrow = L.control({position: 'topleft'});
        north_arrow.onAdd = function(map) {
            var div = L.DomUtil.create('div', 'info north-arrow');
            div.innerHTML = '<div style="transform: rotate(0deg); font-size: 24px;">⬆</div>';
            div.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
            div.style.padding = '6px 8px';
            div.style.border = '2px solid rgba(0,0,0,0.2)';
            div.style.borderRadius = '4px';
            div.style.margin = '10px';
            return div;
        };
        north_arrow.addTo({{this._parent.get_name()}});
        {% endmacro %}
    """)

    m.add_child(north_arrow)

    # We’ll add all data layers now, one per numeric column, and show/hide them via JS radios.
    layer_js_names = {}      # {col: <folium internal JS var name>}
    legend_html_snippets = {}  # {col: HTML string}

    # Tooltip: show ALL fields (not just selected)
    fields = [c for c in gdf.columns if c != gdf.geometry.name]
    # Optional rounding for display
    aliases = []
    tooltip_df = gdf[fields].copy()
    if tooltip_precision is not None:
        for c in fields:
            if pd.api.types.is_numeric_dtype(tooltip_df[c]):
                tooltip_df[c] = tooltip_df[c].round(tooltip_precision)
    # Use aliases as "pretty" labels in the same order
    aliases = [f"{c}:" for c in fields]

    # Pre-serialize GeoJSON once for performance
    geojson_str = gdf.to_json(drop_id=False)

    for col in numeric_columns:
        values = gdf[col].replace([np.inf, -np.inf], np.nan).astype(float)

        vmin = float(np.nanmin(values)) if not np.isnan(values).all() else 0.0
        vmax = float(np.nanmax(values)) if not np.isnan(values).all() else 0.0
        if vmin == vmax:
            # Avoid zero-range colormap: pad a tiny epsilon
            vmax = vmin + 1e-9

        cmap = linear.viridis.scale(vmin, vmax)

        # Legend HTML (one per column; we’ll toggle visibility)
        step = cmap.to_step(index=[vmin + i * (vmax - vmin) / 5 for i in range(6)])
        legend_html = step._repr_html_() if hasattr(step, "_repr_html_") else step.caption

        # Wrap with a container we can toggle by id
        legend_html_snippets[col] = f"""
            <div class="legend-container leaflet-control" id="legend-{col}" style="display:none;">
              <div style="background:white; padding:8px 10px; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,0.3);">
                <div style="font-weight:600; margin-bottom:6px">{col}</div>
                {legend_html}
              </div>
            </div>
        """

        # Style function for the current column
        def _style_factory(vmin=vmin, vmax=vmax, cmap=cmap, col=col):
            def style_fn(feature):
                val = feature["properties"].get(col, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    color = "#cccccc"
                else:
                    try:
                        color = cmap(float(val))
                    except Exception:
                        color = "#cccccc"
                return {
                    "fillColor": color,
                    "color": "#555555",
                    "weight": 0.7,
                    "fillOpacity": 0.8,
                }
            return style_fn

        gj = GeoJson(
            data=json.loads(geojson_str),
            name=f"__data_{col}",  # hidden from layer control; we control via custom radios
            style_function=_style_factory(),
            highlight_function=lambda feat: {"weight": 2, "color": "#111111", "fillOpacity": 0.95},
            tooltip=GeoJsonTooltip(
                fields=fields,
                aliases=aliases,
                localize=True,
                sticky=True,
                labels=True,
            ),
            control=False,  # IMPORTANT: don't put in LayerControl; we handle visibility ourselves
            overlay=True,
            show=False,
        )
        gj.add_to(m)
        layer_js_names[col] = gj.get_name()  # the variable name in the rendered JS

    # Initial selection
    initial = initial or numeric_columns[0]
    if initial not in numeric_columns:
        initial = numeric_columns[0]

    # Inject our radio UI and the show/hide logic + legend switching

    class _RadioController(MacroElement):
        _template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
          var map = {{ this._parent.get_name() }};

          var layers = {
            {% for col, jsname in this.layer_js_names.items() -%}
            "{{ col }}": {{ jsname }}{{ "," if not loop.last else "" }}
            {%- endfor %}
          };

          // Legend root (bottom-right)
          var legendRoot = L.control({position: 'bottomright'});
          legendRoot.onAdd = function() {
            var div = L.DomUtil.create('div');
            L.DomEvent.disableClickPropagation(div);
            div.id = 'legend-root';
            return div;
          };
          legendRoot.addTo(map);

          // Insert all legends (hidden by default)
          var legendRootDiv = document.getElementById('legend-root');
          legendRootDiv.innerHTML = `
            {% for col, html in this.legend_html_snippets.items() -%}
              {{ html | replace("\\n", " ") | safe }}
            {%- endfor %}
          `;

          function showOnly(col) {
            Object.keys(layers).forEach(function(k) {
              if (map.hasLayer(layers[k])) map.removeLayer(layers[k]);
            });
            map.addLayer(layers[col]);

            var els = legendRootDiv.querySelectorAll('.legend-container');
            els.forEach(function(e){ e.style.display = 'none'; });
            var active = document.getElementById('legend-' + col);
            if (active) active.style.display = 'block';
          }

          // Radio UI (top-right)
          var radioCtl = L.control({position: 'topright'});
          radioCtl.onAdd = function() {
            var div = L.DomUtil.create('div', 'leaflet-control-layers leaflet-control');
            div.style.background = 'white';
            div.style.padding = '8px 10px';
            div.style.borderRadius = '6px';
            div.style.boxShadow = '0 1px 4px rgba(0,0,0,0.3)';
            div.innerHTML = `
              <div style="font-weight:600; margin-bottom:6px">Select metric</div>
              {% for col in this.cols -%}
                <label style="display:block; margin-bottom:4px; cursor:pointer;">
                  <input type="radio" name="metric-radio" value="{{ col }}" {{ 'checked' if col == this.initial else '' }}/>
                  <span style="margin-left:6px">{{ col }}</span>
                </label>
              {%- endfor %}
            `;
            L.DomEvent.disableClickPropagation(div);
            return div;
          };
          radioCtl.addTo(map);

          // Add title
          var titleDiv = L.control({position: 'topcenter'});
          titleDiv.onAdd = function() {
            var div = L.DomUtil.create('div');
            div.style.background = 'white';
            div.style.padding = '8px 10px';
            div.style.borderRadius = '6px';
            div.style.boxShadow = '0 1px 4px rgba(0,0,0,0.3)';
            div.id = 'map-title';
            return div;
          };
          titleDiv.addTo(map);

          // Update title function
          function updateTitle(metric) {
            var titleElement = document.getElementById('map-title');
            titleElement.innerHTML = `<h3 style="margin:0">${metric}</h3>`;
          }

          // Wire up change
          var radios = document.getElementsByName('metric-radio');
          [].forEach.call(radios, function(r) {
            r.addEventListener('change', function(ev) {
              if (ev.target.checked) {
                showOnly(ev.target.value);
                updateTitle(ev.target.value);
              }
            });
          });

          // Activate initial and set initial title
          showOnly("{{ this.initial }}");
          updateTitle("{{ this.initial }}");
        })();
        {% endmacro %}
        """)

        def __init__(self, layer_js_names, legend_html_snippets, cols, initial):
            super().__init__()
            self._name = "RadioController"
            self.layer_js_names = layer_js_names
            self.legend_html_snippets = legend_html_snippets
            self.cols = cols
            self.initial = initial

        def render(self, **kwargs):
            super().render(**kwargs)

    m.add_child(
        _RadioController(
            layer_js_names=layer_js_names,
            legend_html_snippets=legend_html_snippets,
            cols=numeric_columns,
            initial=initial,
        )
    )

    # Fit bounds once (so base never disappears) and keep basemap unchanged on switches
    if not np.isnan(bounds).any():
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    if outfile:
        outfile_path = os.path.join(scenario_dir, outfile)
        os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
        m.save(outfile_path)

    return m


# ... existing code ...

def compare_scenarios(
        study_dir: str,
        scenario1_name: str = "Scenario 1",
        scenario2_name: str = "Scenario 2",
        output_dir: str = None,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Compare results between two scenarios.

    Loads results from both scenarios, calculates differences and percent changes,
    prints key summary statistics, and creates detailed comparison files and maps.

    Args:
        scenario1_dir: Path to first scenario directory (baseline)
        scenario2_dir: Path to second scenario directory (alternative)
        scenario1_name: Display name for scenario 1
        scenario2_name: Display name for scenario 2
        output_dir: Optional directory to save comparison results (defaults to scenario2_dir/comparison)

    Returns:
        tuple of (userclass_comparison_df, geometry_comparison_gdf)
    """

    scenario1_dir = os.path.join(study_dir, scenario1_name)
    scenario2_dir = os.path.join(study_dir, scenario2_name)

    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(scenario2_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"SCENARIO COMPARISON")
    print(f"Baseline: {scenario1_name}")
    print(f"Alternative: {scenario2_name}")
    print(f"{'=' * 60}\n")

    # ========================================================================
    # Load userclass results
    # ========================================================================
    print("Loading userclass results...")
    uc1 = pd.read_csv(os.path.join(scenario1_dir, "results", "userclass_results.csv"), index_col=0)
    uc2 = pd.read_csv(os.path.join(scenario2_dir, "results", "userclass_results.csv"), index_col=0)

    # Calculate differences
    uc_comparison = pd.DataFrame(index=uc1.index)

    # Copy scenario 1 values
    for col in uc1.columns:
        uc_comparison[f"{scenario1_name}_{col}"] = uc1[col]

    # Copy scenario 2 values
    for col in uc2.columns:
        uc_comparison[f"{scenario2_name}_{col}"] = uc2[col]

    # Calculate absolute and percent differences for key metrics
    uc_comparison["total_value_diff"] = uc2["total_value"] - uc1["total_value"]
    uc_comparison["total_value_pct_change"] = (
            (uc2["total_value"] - uc1["total_value"]) / uc1["total_value"] * 100
    )

    uc_comparison["per_capita_diff"] = uc2["per_capita"] - uc1["per_capita"]
    uc_comparison["per_capita_pct_change"] = (
            (uc2["per_capita"] - uc1["per_capita"]) / uc1["per_capita"] * 100
    )

    # Mode share differences (percentage points)
    for mode in MODES:
        col = f"percent_from_{mode}"
        if col in uc1.columns and col in uc2.columns:
            uc_comparison[f"{mode}_share_diff_pp"] = (uc2[col] - uc1[col]) * 100  # percentage points

    # Save userclass comparison
    uc_comparison.to_csv(os.path.join(output_dir, "userclass_comparison.csv"))

    # ========================================================================
    # Print overall summary statistics
    # ========================================================================
    total_value_1 = uc1["total_value"].sum()
    total_value_2 = uc2["total_value"].sum()
    total_value_diff = total_value_2 - total_value_1
    total_value_pct_change = (total_value_diff / total_value_1) * 100

    total_pop_1 = uc1["total_pop"].sum()
    total_pop_2 = uc2["total_pop"].sum()

    per_capita_1 = total_value_1 / total_pop_1
    per_capita_2 = total_value_2 / total_pop_2
    per_capita_diff = per_capita_2 - per_capita_1
    per_capita_pct_change = (per_capita_diff / per_capita_1) * 100

    print(f"\n{'=' * 60}")
    print(f"OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"\nTotal Accessibility Value:")
    print(f"  {scenario1_name}: {total_value_1:,.0f}")
    print(f"  {scenario2_name}: {total_value_2:,.0f}")
    print(f"  Difference: {total_value_diff:,.0f} ({total_value_pct_change:+.2f}%)")

    print(f"\nPer Capita Accessibility:")
    print(f"  {scenario1_name}: {per_capita_1:,.2f}")
    print(f"  {scenario2_name}: {per_capita_2:,.2f}")
    print(f"  Difference: {per_capita_diff:,.2f} ({per_capita_pct_change:+.2f}%)")

    # Mode shares
    print(f"\n{'=' * 60}")
    print(f"OVERALL MODE SHARES")
    print(f"{'=' * 60}")
    for mode in MODES:
        col = f"value_from_{mode}"
        if col in uc1.columns and col in uc2.columns:
            mode_value_1 = uc1[col].sum()
            mode_value_2 = uc2[col].sum()
            mode_share_1 = (mode_value_1 / total_value_1) * 100
            mode_share_2 = (mode_value_2 / total_value_2) * 100
            mode_share_diff = mode_share_2 - mode_share_1

            print(f"\n{mode}:")
            print(f"  {scenario1_name}: {mode_share_1:.2f}%")
            print(f"  {scenario2_name}: {mode_share_2:.2f}%")
            print(f"  Difference: {mode_share_diff:+.2f} percentage points")

    # ========================================================================
    # Load geometry results
    # ========================================================================
    print(f"\n{'=' * 60}")
    print("Loading geometry results...")
    geom1 = gpd.read_file(os.path.join(scenario1_dir, "results", "geometry_results.gpkg"))
    geom2 = gpd.read_file(os.path.join(scenario2_dir, "results", "geometry_results.gpkg"))

    # Ensure same index
    geom1 = geom1.set_index(geom1.index.astype(str))
    geom2 = geom2.set_index(geom2.index.astype(str))

    # Create comparison GeoDataFrame
    geom_comparison = geom1[['geometry']].copy()

    # Add scenario values
    for col in ['total_value', 'per_capita', 'total_pop']:
        if col in geom1.columns:
            geom_comparison[f"{scenario1_name}_{col}"] = geom1[col]
            geom_comparison[f"{scenario2_name}_{col}"] = geom2[col]

    # Calculate differences
    geom_comparison["total_value_diff"] = geom2["total_value"] - geom1["total_value"]
    geom_comparison["total_value_pct_change"] = (
            (geom2["total_value"] - geom1["total_value"]) / geom1["total_value"].replace(0, np.nan) * 100
    )

    geom_comparison["per_capita_diff"] = geom2["per_capita"] - geom1["per_capita"]
    geom_comparison["per_capita_pct_change"] = (
            (geom2["per_capita"] - geom1["per_capita"]) / geom1["per_capita"].replace(0, np.nan) * 100
    )

    # Mode share differences
    for mode in MODES:
        col = f"percent_from_{mode}"
        if col in geom1.columns and col in geom2.columns:
            geom_comparison[f"{mode}_share_diff_pp"] = (geom2[col] - geom1[col]) * 100
            geom_comparison[f"{scenario1_name}_{col}"] = geom1[col] * 100
            geom_comparison[f"{scenario2_name}_{col}"] = geom2[col] * 100

    # Save geometry comparison
    geom_comparison.to_file(os.path.join(output_dir, "geometry_comparison.gpkg"), driver="GPKG")

    # ========================================================================
    # Create visualization maps
    # ========================================================================
    print("\nCreating comparison visualizations...")

    # Map 1: Value changes
    value_cols = [
        "total_value_diff",
        "total_value_pct_change",
        "per_capita_diff",
        "per_capita_pct_change",
    ]
    value_cols = [c for c in value_cols if c in geom_comparison.columns]

    make_radio_choropleth_map(
        scenario_dir=output_dir,
        in_data=geom_comparison,
        outfile="value_comparison_map.html",
        columns_to_viz=value_cols,
        initial="per_capita_pct_change",
    )

    # Map 2: Mode share changes
    mode_share_cols = []
    for mode in MODES:
        col = f"{mode}_share_diff_pp"
        if col in geom_comparison.columns:
            mode_share_cols.append(col)

    if mode_share_cols:
        make_radio_choropleth_map(
            scenario_dir=output_dir,
            in_data=geom_comparison,
            outfile="mode_share_comparison_map.html",
            columns_to_viz=mode_share_cols,
            initial=mode_share_cols[0],
        )

    # ========================================================================
    # Print geographic summary
    # ========================================================================
    print(f"\n{'=' * 60}")
    print(f"GEOGRAPHIC DISTRIBUTION")
    print(f"{'=' * 60}")

    # Areas with biggest gains/losses
    top_gainers = geom_comparison.nlargest(5, "per_capita_pct_change")
    top_losers = geom_comparison.nsmallest(5, "per_capita_pct_change")

    print(f"\nTop 5 areas with largest per-capita gains:")
    for idx in top_gainers.index:
        val = top_gainers.loc[idx, "per_capita_pct_change"]
        print(f"  Zone {idx}: {val:+.2f}%")

    print(f"\nTop 5 areas with largest per-capita losses:")
    for idx in top_losers.index:
        val = top_losers.loc[idx, "per_capita_pct_change"]
        print(f"  Zone {idx}: {val:+.2f}%")

    # ========================================================================
    # Summary file
    # ========================================================================
    with open(os.path.join(output_dir, "comparison_summary.txt"), "w") as f:
        f.write(f"SCENARIO COMPARISON SUMMARY\n")
        f.write(f"={'-'*60}\n\n")
        f.write(f"Baseline: {scenario1_name}\n")
        f.write(f"Alternative: {scenario2_name}\n\n")

        f.write(f"OVERALL RESULTS\n")
        f.write(f"{'-' * 60}\n")
        f.write(f"Total Accessibility Value:\n")
        f.write(f"  {scenario1_name}: {total_value_1:,.0f}\n")
        f.write(f"  {scenario2_name}: {total_value_2:,.0f}\n")
        f.write(f"  Difference: {total_value_diff:,.0f} ({total_value_pct_change:+.2f}%)\n\n")

        f.write(f"Per Capita Accessibility:\n")
        f.write(f"  {scenario1_name}: {per_capita_1:,.2f}\n")
        f.write(f"  {scenario2_name}: {per_capita_2:,.2f}\n")
        f.write(f"  Difference: {per_capita_diff:,.2f} ({per_capita_pct_change:+.2f}%)\n\n")

        f.write(f"MODE SHARES\n")
        f.write(f"{'-' * 60}\n")
        for mode in MODES:
            col = f"value_from_{mode}"
            if col in uc1.columns and col in uc2.columns:
                mode_value_1 = uc1[col].sum()
                mode_value_2 = uc2[col].sum()
                mode_share_1 = (mode_value_1 / total_value_1) * 100
                mode_share_2 = (mode_value_2 / total_value_2) * 100
                mode_share_diff = mode_share_2 - mode_share_1

                f.write(f"\n{mode}:\n")
                f.write(f"  {scenario1_name}: {mode_share_1:.2f}%\n")
                f.write(f"  {scenario2_name}: {mode_share_2:.2f}%\n")
                f.write(f"  Difference: {mode_share_diff:+.2f} percentage points\n")

        print(f"\n{'=' * 60}")
        print(f"Comparison complete!")
        print(f"Results saved to: {output_dir}")
        print(f"  - userclass_comparison.csv")
        print(f"  - geometry_comparison.gpkg")
        print(f"  - value_comparison_map.html")
        print(f"  - mode_share_comparison_map.html")
        print(f"  - comparison_summary.txt")
        print(f"{'=' * 60}\n")

    return uc_comparison, geom_comparison
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


def visualize_access_to_zone(geoms_with_dests,
                             ttm,
                             target_zone_id = None,
                             out_path = "communication/access_to_zone.html"):
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

    def style_function(feature):
        # Highlight the target zone in blue
        if str(feature['properties']['geom_id']) == target_zone_id:
            return {'fillColor': 'blue', 'color': 'blue', 'weight': 2}
        return {}

    m = to_visualize.explore(
        column="travel_time_to_dest",
        cmap="YlOrBr",
        tiles="CartoDB.Positron",
        legend=True,
        legend_kwds={"caption": "Travel Time to Selected Destination (minutes)"},
        style_kwds={"style_function":style_function}
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save(out_path)
    print(
        f"Saved visualization of access to zone {target_zone_id} to {out_path}"
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


def make_radio_choropleth_map(
        scenario_dir: str,
        in_data: gpd.GeoDataFrame | str,
        outfile: str | None = None,
        numeric_columns: list[str] | None = None,
        initial: str | None = None,
        tiles: str = "CartoDB positron",
        tooltip_precision: int | None = None,
        exclude: list = [],
) -> folium.Map:
    """
    Build a Folium map where the user can choose exactly one numeric column
    (via radio buttons) to color the layer. The basemap never changes, and
    exactly one legend is shown at any time.

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

    # 2) Determine numeric columns
    auto_numeric = [c for c in gdf.columns if c != gdf.geometry.name and pd.api.types.is_numeric_dtype(gdf[c])]
    if numeric_columns is None:
        numeric_columns = auto_numeric
    else:
        # keep only those that are actually numeric after coercion
        numeric_columns = [c for c in numeric_columns if c in auto_numeric]

    if not numeric_columns:
        raise ValueError("No numeric columns available to visualize after coercion.")

    # 3) Build the base map (basemap never changes)
    # Center on the data bounds
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    cx = (bounds[1] + bounds[3]) / 2
    cy = (bounds[0] + bounds[2]) / 2
    m = folium.Map(location=[cx, cy], zoom_start=12, tiles=tiles, control_scale=True)

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
    from folium.elements import MacroElement
    from jinja2 import Template

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

          // Wire up change
          var radios = document.getElementsByName('metric-radio');
          [].forEach.call(radios, function(r) {
            r.addEventListener('change', function(ev) {
              if (ev.target.checked) showOnly(ev.target.value);
            });
          });

          // Activate initial
          showOnly("{{ this.initial }}");
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


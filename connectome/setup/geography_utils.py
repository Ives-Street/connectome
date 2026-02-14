# geography_utils.py
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import geopandas as gpd
import pandas as pd

import logging
from tqdm import tqdm

# This requires the MAUP library: https://github.com/spatialucr/maup
# Make sure it's installed in your environment.
try:
    import maup  # type: ignore
except ImportError:
    maup = None

logger = logging.getLogger(__name__)


@dataclass
class ArealInterpolationConfig:
    """
    Configuration object for areal interpolation (change of spatial support)
    from source geometries (e.g., tracts) to target geometries (e.g., TAZs).

    Attributes
    ----------
    source_key : str
        Column in both `gdf_source` and `df_source_data` that joins geometries
        to attribute data (e.g., 'geom_id' for tracts).
    target_key : str
        Column in `gdf_target` that uniquely identifies target geometries
        (e.g., 'geom_id' for TAZs).
    group_cols : Sequence[str]
        Columns in `df_source_data` that define categories that should be
        preserved, not summed away (e.g. 'user_class_id', 'race', etc.).
        The interpolation runs within each unique combination of group_cols.
    value_cols : Optional[Sequence[str]]
        Columns to be interpolated (typically numeric). If None, uses all
        numeric columns in `df_source_data` except group_cols and source_key.
    """
    source_key: str = "geom_id"
    target_key: str = "geom_id"
    group_cols: Sequence[str] = ()
    value_cols: Optional[Sequence[str]] = None


def _prepare_value_columns(df: pd.DataFrame, config: ArealInterpolationConfig) -> List[str]:
    """Infer value columns if not specified in config."""
    if config.value_cols is not None:
        return list(config.value_cols)

    # Default: all numeric columns that are not in group_cols or key
    exclude = set(config.group_cols) | {config.source_key}
    numeric_cols = df.select_dtypes(include="number").columns
    value_cols = [c for c in numeric_cols if c not in exclude]
    if not value_cols:
        raise ValueError(
            "No numeric columns found to interpolate. "
            "Specify `value_cols` in ArealInterpolationConfig."
        )
    return value_cols


def compute_areal_weights(
    gdf_source: gpd.GeoDataFrame,
    gdf_target: gpd.GeoDataFrame,
    equal_area_crs: str = "ESRI:54009",
):
    """
    Compute area-based interpolation weights from source to target geometries.

    If a modern MAUP with `weights` is available, use it.
    Otherwise, fall back to a GeoPandas-based implementation.
    """
    logger.info(
        "Computing areal weights: %d source geometries, %d target geometries",
        len(gdf_source),
        len(gdf_target),
    )

    # Ensure both are in same CRS, then project to equal-area CRS for area-based weights
    if gdf_source.crs is None or gdf_target.crs is None:
        raise ValueError("Both gdf_source and gdf_target must have a CRS defined.")

    if gdf_source.crs != gdf_target.crs:
        gdf_target = gdf_target.to_crs(gdf_source.crs)

    gdf_source_proj = gdf_source.to_crs(equal_area_crs).copy()
    gdf_target_proj = gdf_target.to_crs(equal_area_crs).copy()

    # --- Fast path: use MAUP if it has `weights` ---
    if (maup is not None) and hasattr(maup, "weights"):
        logger.info("Using MAUP fast path for areal weights (maup.weights).")
        # Aligning geometries if `align` exists, otherwise call weights directly
        if hasattr(maup, "align"):
            logger.info("Aligning geometries with maup.align before computing weights.")
            src_aligned, tgt_aligned = maup.align(gdf_source_proj, gdf_target_proj)  # type: ignore[attr-defined]
            weights = maup.weights(src_aligned, tgt_aligned)
        else:
            weights = maup.weights(gdf_source_proj, gdf_target_proj)
        logger.info("Finished computing MAUP weights.")
        return weights

    # --- Fallback: pure GeoPandas-based areal weights ---
    logger.info("MAUP weights not available; using GeoPandas overlay fallback.")

    # Preserve original indices as columns
    gdf_source_proj = gdf_source_proj.reset_index().rename(columns={"index": "src_idx"})
    gdf_target_proj = gdf_target_proj.reset_index().rename(columns={"index": "tgt_idx"})

    logger.info("Performing overlay (intersection) of source and target geometries...")
    intersections = gpd.overlay(
        gdf_source_proj[["src_idx", "geometry"]],
        gdf_target_proj[["tgt_idx", "geometry"]],
        how="intersection",
    )

    if intersections.empty:
        raise ValueError("No intersections found between source and target geometries.")
    logger.info("Overlay produced %d intersection polygons.", len(intersections))

    # Compute intersection areas
    intersections["intersect_area"] = intersections.geometry.area

    # Total area per source geometry
    src_total_area = (
        intersections.groupby("src_idx")["intersect_area"]
        .sum()
        .rename("src_total_area")
    )
    intersections = intersections.merge(src_total_area, on="src_idx", how="left")

    # Weight = intersection area / total area of that source
    intersections["weight"] = intersections["intersect_area"] / intersections["src_total_area"]

    # Return as a DataFrame keyed by src_idx / tgt_idx
    weights_df = intersections[["src_idx", "tgt_idx", "weight"]].copy()
    logger.info("Finished computing fallback areal weights.")
    return weights_df


def interpolate_attributes(
        gdf_source: Union[gpd.GeoDataFrame, str],
        df_source_data: Union[pd.DataFrame, str],
        gdf_target: Union[gpd.GeoDataFrame, str],
        config: ArealInterpolationConfig,
    equal_area_crs: str = "ESRI:54009",
) -> pd.DataFrame:
    """
    Perform area-weighted areal interpolation of attribute data
    from source geometries to target geometries.

    Typical use in this project:
    - source = census-tract geometries
    - df_source_data = userclass_statistics by tract
    - target = TAZ geometries
    - group_cols = ['user_class_id', 'race', 'hispanic_or_latino', 'car_owner']
    - value_cols = ['population']

    Parameters
    ----------
    gdf_source : GeoDataFrame or str
        Source geometries keyed by `config.source_key`. If str, path to file to read.
    df_source_data : DataFrame or str
        Attribute data keyed by `config.source_key`. May contain multiple rows
        per source geometry, differentiated by `group_cols`. If str, path to file to read.
    gdf_target : GeoDataFrame or str
        Target geometries keyed by `config.target_key`. If str, path to file to read.
    config : ArealInterpolationConfig
        Configuration for keys, grouping, and value columns.
    equal_area_crs : str, default 'ESRI:54009'
        CRS for area computations.

    Returns
    -------
    pd.DataFrame
        Interpolated attributes at the target level, with columns:
        - config.target_key (e.g., TAZ geom_id)
        - all group_cols
        - all value_cols, now representing totals within each target cell
          and group combination.
    """
    # Read input files if paths provided
    if isinstance(gdf_source, str):
        gdf_source = gpd.read_file(gdf_source)
    if isinstance(df_source_data, str):
        df_source_data = pd.read_csv(df_source_data)
    if isinstance(gdf_target, str):
        gdf_target = gpd.read_file(gdf_target)

    if config.source_key not in gdf_source.columns:
        raise KeyError(f"Source key '{config.source_key}' not found in gdf_source columns.")

    if config.source_key not in df_source_data.columns:
        raise KeyError(f"Source key '{config.source_key}' not found in df_source_data columns.")

    if config.target_key not in gdf_target.columns:
        raise KeyError(f"Target key '{config.target_key}' not found in gdf_target columns.")

    # Ensure join key types match between source geometries and attribute data
    if gdf_source[config.source_key].dtype != df_source_data[config.source_key].dtype:
        gdf_source[config.source_key] = gdf_source[config.source_key].astype(str)
        df_source_data[config.source_key] = df_source_data[config.source_key].astype(str)

    # Prepare value columns
    value_cols = _prepare_value_columns(df_source_data, config)

    # Merge geometry into attribute data
    df = df_source_data.merge(
        gdf_source[[config.source_key, "geometry"]],
        on=config.source_key,
        how="left",
        validate="m:1",
    )
    if df["geometry"].isna().any():
        raise ValueError("Some source rows in df_source_data do not have matching geometries in gdf_source.")

    gdf_src_full = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf_source.crs)

    # Compute areal weights once across all groups
    weights = compute_areal_weights(gdf_src_full, gdf_target, equal_area_crs=equal_area_crs)

    # Create a mapping from target index back to target_key
    target_index_to_key = gdf_target[config.target_key]

    group_cols = list(config.group_cols)

    # --- Fast path: MAUP weights + distribute (if available and weights is a Series) ---
    if (maup is not None) and hasattr(maup, "distribute") and isinstance(weights, pd.Series):
        if group_cols:
            grouped = gdf_src_full.groupby(group_cols, dropna=False)
        else:
            grouped = [((), gdf_src_full)]

        interpolated_frames: List[pd.DataFrame] = []

        for group_values, gdf_group in grouped:
            mask = gdf_src_full.index.isin(gdf_group.index)
            weights_sub = weights[mask]

            distributed_cols = {}
            for col in value_cols:
                series_src = gdf_src_full.loc[mask, col]
                distributed = maup.distribute(weights_sub, series_src)
                distributed_cols[col] = distributed

            df_target_group = pd.DataFrame(distributed_cols)
            df_target_group[config.target_key] = target_index_to_key

            if group_cols:
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                for col_name, col_val in zip(group_cols, group_values):
                    df_target_group[col_name] = col_val

            interpolated_frames.append(df_target_group)

        result = pd.concat(interpolated_frames, axis=0)

        group_by_cols = [config.target_key] + list(group_cols)
        result = (
            result.groupby(group_by_cols, dropna=False)[value_cols]
            .sum()
            .reset_index()
        )
        return result

    # --- Fallback path: GeoPandas-based weights DataFrame ---
    # Expect columns: src_idx, tgt_idx, weight
    if not isinstance(weights, pd.DataFrame) or not {"src_idx", "tgt_idx", "weight"} <= set(weights.columns):
        raise TypeError(
            "Fallback interpolation expects weights as a DataFrame with "
            "columns {'src_idx', 'tgt_idx', 'weight'}."
        )

    # Attach a stable source index to match weights.src_idx
    src_with_idx = gdf_src_full.reset_index().rename(columns={"index": "src_idx"})
    src_needed_cols = ["src_idx"] + group_cols + list(value_cols)
    src_attrs = src_with_idx[src_needed_cols]

    # Map target indices to target_key values
    tgt_index_to_key = (
        target_index_to_key
        .reset_index()
        .rename(columns={"index": "tgt_idx", config.target_key: "target_id"})
    )

    # Join weights with source attributes and target IDs
    weights_with_attrs = weights.merge(src_attrs, on="src_idx", how="left")
    weights_with_attrs = weights_with_attrs.merge(tgt_index_to_key, on="tgt_idx", how="left")

    # Apply weights to each value column
    for col in value_cols:
        weights_with_attrs[col] = weights_with_attrs[col] * weights_with_attrs["weight"]

    # Group by target and group_cols, summing contributions
    group_by_cols = ["target_id"] + group_cols if group_cols else ["target_id"]
    result = (
        weights_with_attrs
        .groupby(group_by_cols, dropna=False)[value_cols]
        .sum()
        .reset_index()
        .rename(columns={"target_id": config.target_key})
    )

    return result


def interpolate_tracts_to_tazs(
        tracts: Union[gpd.GeoDataFrame, str],
        tazs: Union[gpd.GeoDataFrame, str],
        userclass_statistics: Union[pd.DataFrame, str],
        taz_id_col: str = "geom_id",
        create_taz_id_col: bool = True,
        source_geom_id_col: str = "geom_id",
        save_userclass_csv_to: Optional[str] = None,
        save_analysis_areas_gpkg_to: Optional[str] = None,
        interpolate_tract_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    High-level helper tailored to this project:
    Interpolate tract-level `userclass_statistics` onto TAZs (or other polygons).

    Parameters
    ----------
    tracts : GeoDataFrame or str
        Census tract geometries or path to file containing them.
        Must have `source_geom_id_col` and `geometry`.
    tazs : GeoDataFrame or str
        Target geometries (TAZs or arbitrary polygons) or path to file containing them.
        Must have `taz_id_col` and `geometry`.
    userclass_statistics : DataFrame or str
        Tract-level userclass statistics as produced by `create_userclass_statistics` 
        or path to CSV file containing them. Must have:
          - source_geom_id_col
          - 'user_class_id'
          - 'race'
          - 'hispanic_or_latino'
          - 'car_owner'
          - 'population'
    taz_id_col : str, default 'geom_id'
        Column in `tazs` that uniquely identifies each TAZ.
    create_taz_id_col: bool = True,
        Create the ID column with unique integers, overwrites taz_id_col if provided.
    source_geom_id_col : str, default 'geom_id'
        Column in `tracts` and `userclass_statistics` that identifies source geometries.
    save_userclass_csv_to : Optional[str]
        If provided, save the interpolated userclass result to this CSV file.
    save_analysis_areas_gpkg_to : Optional[str] 
        If provided, save the (possibly modified) TAZ dataframe to this gpkg file.
    interpolate_tract_cols: Optional[list[str]]
        If provided, interpolate these additional columns from the tracts dataframe
        onto the TAZs.

    Returns
    -------
    pd.DataFrame
        TAZ-level userclass statistics with same schema as input userclass_statistics
        except that `geom_id` now refers to TAZ ids and populations have been
        re-allocated accordingly.
    """
    # Read input files if paths provided 
    if isinstance(tracts, str):
        tracts = gpd.read_file(tracts)
    if isinstance(tazs, str):
        tazs = gpd.read_file(tazs)
    if isinstance(userclass_statistics, str):
        userclass_statistics = pd.read_csv(userclass_statistics)

    required_cols = {
        source_geom_id_col,
        "user_class_id",
        "race",
        "hispanic_or_latino",
        "car_owner",
        "population",
    }
    missing = required_cols - set(userclass_statistics.columns)
    if missing:
        raise KeyError(f"userclass_statistics is missing required columns: {missing}")

    if source_geom_id_col not in tracts.columns:
        raise KeyError(f"'{source_geom_id_col}' not found in tracts columns.")

    if create_taz_id_col:  # create the ID column
        tazs[taz_id_col] = range(1, len(tazs) + 1)

    if taz_id_col not in tazs.columns:
        raise KeyError(f"'{taz_id_col}' not found in tazs columns.")

    config = ArealInterpolationConfig(
        source_key=source_geom_id_col,
        target_key=taz_id_col,
        group_cols=("user_class_id", "race", "hispanic_or_latino", "car_owner"),
        value_cols=("population",),
    )

    interpolated = interpolate_attributes(
        gdf_source=tracts,
        df_source_data=userclass_statistics,
        gdf_target=tazs,
        config=config,
    )

    # Also interpolate any additional tract columns if specified
    if interpolate_tract_cols:
        available_cols = [col for col in interpolate_tract_cols if col in tracts.columns]
        if len(available_cols) != len(interpolate_tract_cols):
            missing_cols = set(interpolate_tract_cols) - set(tracts.columns)
            logger.warning(f"Some requested tract columns were not found and will be skipped: {missing_cols}")
        interpolate_tract_cols = available_cols
        tract_data = tracts[[source_geom_id_col] + interpolate_tract_cols].copy()
        tract_config = ArealInterpolationConfig(
            source_key=source_geom_id_col,
            target_key=taz_id_col,
            value_cols=interpolate_tract_cols
        )

        interpolated_tract_cols = interpolate_attributes(
            gdf_source=tracts,
            df_source_data=tract_data,
            gdf_target=tazs,
            config=tract_config
        )

        # Merge the interpolated tract columns into the tazs geodataframe
        tazs = tazs.merge(
            interpolated_tract_cols,
            on=taz_id_col,
            how='left'
        )

    # For consistency with the tract-level schema, we want the output column
    # still to be called 'geom_id' if taz_id_col is something else./
    if taz_id_col != "geom_id":
        interpolated = interpolated.rename(columns={taz_id_col: "geom_id"})

    if save_userclass_csv_to:
        interpolated.to_csv(save_userclass_csv_to, index=False)

    if save_analysis_areas_gpkg_to:
        if os.path.exists(save_analysis_areas_gpkg_to):
            os.remove(save_analysis_areas_gpkg_to)
        tazs.to_file(save_analysis_areas_gpkg_to, driver="GPKG")


    # Assert that total population is preserved within a small tolerance
    original_total = userclass_statistics["population"].sum()
    interpolated_total = interpolated["population"].sum()
    assert abs(original_total - interpolated_total) < 1, (
        f"Total population not preserved: original={original_total}, interpolated={interpolated_total}"
    )

    return interpolated


def calculate_population_per_sqkm(input_dir: str, save_file: bool = True):
    """
    Compute population density (population per square kilometer) for analysis areas.

    - Ensures that an 'area_sqm' column exists; if not, it is computed from geometry.
    - Joins tract-level population using consistent string-based geom_id keys.
    - Optionally overwrites the existing analysis_areas.gpkg with the new columns.
    """
    analysis_areas = gpd.read_file(f"{input_dir}/analysis_areas.gpkg")

    # Ensure we have an area_sqm column; if not, compute it in a projected CRS
    if "area_sqm" not in analysis_areas.columns or analysis_areas["area_sqm"].isna().any():
        if analysis_areas.crs is None:
            raise ValueError(
                "analysis_areas has no CRS set. Cannot compute area_sqm reliably. "
                "Please set a CRS on analysis_areas.gpkg."
            )

        # Use an appropriate projected CRS for area calculation
        if analysis_areas.crs.is_geographic:
            analysis_areas_proj = analysis_areas.to_crs(analysis_areas.estimate_utm_crs())
        else:
            analysis_areas_proj = analysis_areas

        analysis_areas["area_sqm"] = analysis_areas_proj.geometry.area

    if "population_per_sqkm" not in analysis_areas.columns:
        logger.info("calculating population per sqkm")

        userclass_statistics = pd.read_csv(f"{input_dir}/userclass_statistics.csv")

        # Use consistent string keys for joining
        userclass_statistics["geom_id"] = userclass_statistics["geom_id"].astype(str)
        analysis_areas["geom_id"] = analysis_areas["geom_id"].astype(str)

        population_by_geoid = userclass_statistics.groupby("geom_id")["population"].sum()

        analysis_areas["census_total_pop"] = analysis_areas["geom_id"].map(population_by_geoid)

        # Avoid division by zero / NaN areas
        valid_area_mask = analysis_areas["area_sqm"] > 0
        analysis_areas["population_per_sqkm"] = pd.NA
        analysis_areas.loc[valid_area_mask, "population_per_sqkm"] = (
            analysis_areas.loc[valid_area_mask, "census_total_pop"]
            / (analysis_areas.loc[valid_area_mask, "area_sqm"] / 1_000_000)
        )

        if os.path.exists(f"{input_dir}/analysis_areas.gpkg"):
            os.remove(f"{input_dir}/analysis_areas.gpkg")

        if save_file:
            analysis_areas.to_file(f"{input_dir}/analysis_areas.gpkg", driver="GPKG")
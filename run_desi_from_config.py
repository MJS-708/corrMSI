# run_desi_from_config.py
from __future__ import annotations

import time

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
import spatialdata as sd

from spatialdata import SpatialData

from desi_spatial_helpers import (
    get_cs_names,
    find_desi_tables,
    find_grid_shapes,
    find_roi_label_shapes,
    annotate_pixels_with_roi_strtree,
    find_shapes_by_regex,
    build_pixel_to_cell_radius_map_spatialdata, 
    make_roi_binary_columns,
    add_roi_distance_columns,
    qc_plot_roi_overlay,
    qc_plot_cell_overlay,
    

    make_sample_sets,
    make_pairs,
    make_upper_triangle_pairs,
    spearman_pairs,
    pearson_pairs,
    moran_univariate_per_sample,
    moran_bivariate_per_sample,
    combine_moran_effects,
    add_fdr_columns,

    # weights config for local/global moran

    local_moran_univariate_summary_per_sample,
    local_moran_bivariate_summary_per_sample,

    # cell radius mapping from centroid+area table
    find_table_by_regex,
    pick_first_existing_col_df,
    build_pixel_to_cell_radius_map_from_table,
    pixel_celltype_features,
    add_cell_halo_columns,
    validate_halo_radius,
    validate_cell_coordinate_space,
)

try:
    import geopandas as gpd
except Exception:
    gpd = None


def _cfg(d: Dict[str, Any], path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _resolve_local_pairs(
    all_pairs: List[Tuple[str, str]],
    subset: Optional[List[List[str]]],
    label: str,
) -> List[Tuple[str, str]]:
    """Filter pairs for local Moran based on an optional YAML pair_subset list.

    subset=None  -> return all_pairs unchanged
    subset=[["A","B"], ...]  -> return only those pairs (both orderings accepted);
                                warns if a requested pair is not in all_pairs.
    """
    if subset is None:
        return all_pairs
    all_set = set(all_pairs)
    resolved = []
    for p in subset:
        fwd = tuple(p)
        rev = (p[1], p[0])
        if fwd in all_set:
            resolved.append(fwd)
        elif rev in all_set:
            resolved.append(rev)
        else:
            print(f"[local_bv] WARNING ({label}): pair {list(p)} not found in global pairs — skipping")
    if not resolved:
        print(f"[local_bv] WARNING ({label}): pair_subset produced no valid pairs — skipping block")
    return resolved


def _find_table_with_cols(sdata: "SpatialData", wanted_cols: Sequence[str]):
    """Return (table_name, matched_col) for the first table that contains any of wanted_cols."""
    for tname, ad in sdata.tables.items():
        cols = list(ad.obs.columns)
        hit = next((c for c in wanted_cols if c in cols), None)
        if hit is not None:
            return tname, hit
    return None, None


def _strip_grid_suffix(grid_name: str, suffix_candidates: Sequence[str]) -> str:
    """Strip the first matching grid suffix from grid_name to recover the sample name."""
    for suf in suffix_candidates:
        if grid_name.endswith(suf):
            return grid_name[: -len(suf)]
    return grid_name


def run_from_yaml(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    zarr_path = _cfg(cfg, "io.zarr_path")
    out_dir = _cfg(cfg, "io.out_dir")
    os.makedirs(out_dir, exist_ok=True)
    
    # QC outputs folder
    qc_dir = os.path.join(out_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    # -----------------------
    # Simple progress counter for long runs
    # -----------------------
    t0 = time.time()
    step = [0]
    def _step(msg: str):
        step[0] += 1
        elapsed = time.time() - t0
        print(f"[step {step[0]}] {msg} (elapsed {elapsed/60.0:.1f} min)")

    # detection patterns
    grid_suffix_candidates = _cfg(cfg, "detect.grid_suffix_candidates", ["_DESI_NORM_grid", "_DESI_grid", "_grid"])
    roi_suffix_candidates = _cfg(cfg, "detect.roi_suffix_candidates", ["_anatomical_labels", "_roi", "_labels"])
    desi_table_regex = _cfg(cfg, "detect.desi_table_regex", ["DESI.*_obs$", ".*DESI.*"])
    # features
    metabolite_subset = _cfg(cfg, "features.metabolite_subset", None)
    max_desi_desi_pairs = _cfg(cfg, "features.max_desi_desi_pairs", 5000)

    # ROI
    join_roi = bool(_cfg(cfg, "roi.join_roi", True))
    write_pixel_table = bool(_cfg(cfg, "roi.write_pixel_table", True))
    roi_bin_prefix = _cfg(cfg, "roi.roi_binary_prefix", "ROI__")
    roi_halo_enabled  = bool(_cfg(cfg, "roi.distance_halo.enabled", False))
    roi_halo_radius   = float(_cfg(cfg, "roi.distance_halo.radius", 10.0))
    roi_halo_decay    = str(_cfg(cfg, "roi.distance_halo.decay", "linear"))
    roi_halo_suffix   = str(_cfg(cfg, "roi.distance_halo.suffix", "_decay"))
    physical_pixel_size_um = float(_cfg(cfg, "roi.distance_halo.physical_pixel_size_um", 100.0))
    roi_label_col = _cfg(cfg, "roi.label_col", None)
    roi_subset = _cfg(cfg, "roi.roi_subset", None)  # None = all ROIs; list = only these labels
    # Regex with one capture group to extract a group name from the full ROI label.
    # e.g. "(?:Ctrl_|HDM_)?([A-Za-z]+)" maps "Ctrl_bronch_01" -> "bronch", "HDM_Bronch_06" -> "Bronch"
    # Set to null to use the full ROI label as-is (one binary column per ROI).
    roi_group_regex = _cfg(cfg, "roi.roi_group_regex", None)

    # cell

    cell_enabled = bool(_cfg(cfg, "cell.enabled", True))
    cell_radius_enabled = bool(_cfg(cfg, "cell.radius_assignment.enabled", False))
    cell_radius_scalar = float(_cfg(cfg, "cell.radius_assignment.cell_radius_scalar", 1.0))
    crop_to_cell_extent = bool(_cfg(cfg, "cell.crop_pixels_to_cell_extent", True))
    cell_coord_system = _cfg(cfg, "cell.coordinate_system", None)  # None = use raw grid units (default); set to sample name if cells are in transformed space
    cell_radius_mode = _cfg(cfg, "cell.mode", "spatialdata")  # spatialdata | table
    cell_shapes_name_regex = _cfg(cfg, "cell.shapes.name_regex", r"cell_types_cells|cell_types|cells")
    cell_rm_types = _cfg(cfg, "cell.rm_cell_types", ["Unidentified"])
    cell_feat_mode = _cfg(cfg, "cell.radius_assignment.pixel_feature_mode", "onehot_any")
    cell_prefix = _cfg(cfg, "cell.radius_assignment.prefix", "CELL__")
    write_cell_meta_csv = bool(_cfg(cfg, "cell.radius_assignment.write_cell_meta_csv", True))
    write_pixel_cell_map_csv = bool(_cfg(cfg, "cell.radius_assignment.write_pixel_to_cell_map_csv", True))
    cell_halo_enabled = bool(_cfg(cfg, "cell.radius_assignment.halo.enabled", False))
    cell_halo_radius  = float(_cfg(cfg, "cell.radius_assignment.halo.radius", 10.0))
    cell_halo_decay   = str(_cfg(cfg, "cell.radius_assignment.halo.decay", "linear"))
    cell_halo_suffix  = str(_cfg(cfg, "cell.radius_assignment.halo.suffix", "_halo"))
    # Used only in legacy table mode; key lives at cell.table_name_regex (not cell.table.name_regex)
    cell_table_name_regex = _cfg(cfg, "cell.table_name_regex", "cell|cells")
    
    cell_x_cands = _cfg(cfg, "cell.table.x_col_candidates", ["x", "cell_x", "centroid_x"])
    cell_y_cands = _cfg(cfg, "cell.table.y_col_candidates", ["y", "cell_y", "centroid_y"])
    cell_area_cands = _cfg(cfg, "cell.table.area_col_candidates", ["area", "cell_area"])
    cell_type_cands = _cfg(cfg, "cell.table.type_col_candidates", ["cell_type", "celltype", "type", "annotation", "label"])


    # weights
    threshold = _cfg(cfg, "weights.threshold", None)
    binary = bool(_cfg(cfg, "weights.binary", True))
    wtype = _cfg(cfg, "weights.type", "distanceband")  # distanceband | knn
    thr_mult = float(_cfg(cfg, "weights.threshold_multiplier", 1.0))
    knn_k = int(_cfg(cfg, "weights.knn_k", 8))
    # Weights cache: set weights.cache_dir to a path to persist weights to disk.
    # The in-process memory cache is always active regardless of this setting.
    _weights_cache_dir = _cfg(cfg, "weights.cache_dir", None)
    weights_cache_dir = os.path.join(out_dir, ".weights_cache") if _weights_cache_dir == "auto"         else (_weights_cache_dir if _weights_cache_dir else None)

    # moran
    moran_enabled = bool(_cfg(cfg, "moran.enabled", True))
    perm_enabled = bool(_cfg(cfg, "moran.permutations.enabled", False))
    perm_n = int(_cfg(cfg, "moran.permutations.n", 999)) if perm_enabled else None
    use_p_sim_for_stats = bool(_cfg(cfg, "moran.permutations.use_p_sim_for_stats", True))
    # outputs: support both old keys (moran.outputs.*) and new nested keys (moran.outputs.global.*)
    moran_out_uni = bool(_cfg(cfg, "moran.outputs.global.univariate", _cfg(cfg, "moran.outputs.univariate", True)))
    moran_out_dd = bool(_cfg(cfg, "moran.outputs.global.bivariate_desi_desi", _cfg(cfg, "moran.outputs.bivariate_desi_desi", True)))
    moran_out_dc = bool(_cfg(cfg, "moran.outputs.global.bivariate_desi_cell", _cfg(cfg, "moran.outputs.bivariate_desi_cell", True)))
    moran_out_dr = bool(_cfg(cfg, "moran.outputs.global.bivariate_desi_roi", _cfg(cfg, "moran.outputs.bivariate_desi_roi", True)))

    local_enabled = bool(_cfg(cfg, "moran.outputs.local.enabled", False))
    local_write_pixel_maps = bool(_cfg(cfg, "moran.outputs.local.write_pixel_maps", False))
    local_alpha = float(_cfg(cfg, "moran.outputs.local.alpha", 0.05))
    local_fdr_on_p = bool(_cfg(cfg, "moran.outputs.local.fdr_on_local_p", True))
    local_p_source = _cfg(cfg, "moran.outputs.local.p_source", "auto")

    # Local bivariate sub-options — each type has enabled + optional pair_subset
    # Supports both old-style (bare if pairs_desi_*) and new nested config.
    # Old behaviour preserved: if the new keys are absent, defaults match old logic.
    local_bv_dd_enabled  = bool(_cfg(cfg, "moran.outputs.local.bivariate_desi_desi.enabled",  False))
    local_bv_dc_enabled  = bool(_cfg(cfg, "moran.outputs.local.bivariate_desi_cell.enabled",  False))
    local_bv_dr_enabled  = bool(_cfg(cfg, "moran.outputs.local.bivariate_desi_roi.enabled",   True))
    local_bv_dd_subset   = _cfg(cfg, "moran.outputs.local.bivariate_desi_desi.pair_subset",   None)
    local_bv_dc_subset   = _cfg(cfg, "moran.outputs.local.bivariate_desi_cell.pair_subset",   None)
    local_bv_dr_subset   = _cfg(cfg, "moran.outputs.local.bivariate_desi_roi.pair_subset",    None)

    combine_enabled = bool(_cfg(cfg, "moran.combine.enabled", True))
    stouffer_weight = _cfg(cfg, "moran.combine.stouffer_weight", "sqrt_n")

    moran_fdr_enabled = bool(_cfg(cfg, "moran.fdr.enabled", True))
    moran_fdr_cols = _cfg(cfg, "moran.fdr.p_columns", ["p_norm", "p_sim", "p_fisher", "p_stouffer", "p_random_effects"])

    # spearman
    spearman_enabled = bool(_cfg(cfg, "spearman.enabled", True))
    sp_out_dd = bool(_cfg(cfg, "spearman.outputs.desi_desi", True))
    sp_out_dc = bool(_cfg(cfg, "spearman.outputs.desi_cell", True))
    sp_out_dr = bool(_cfg(cfg, "spearman.outputs.desi_roi", True))

    sp_mode = _cfg(cfg, "spearman.sample_sets.mode", "each+all")
    sp_custom = _cfg(cfg, "spearman.sample_sets.custom", None)

    sp_fdr_enabled = bool(_cfg(cfg, "spearman.fdr.enabled", True))
    sp_fdr_cols = _cfg(cfg, "spearman.fdr.p_columns", ["p"])

    pearson_enabled = bool(_cfg(cfg, "pearson.enabled", False))
    pe_out_dd = bool(_cfg(cfg, "pearson.outputs.desi_desi", True))
    pe_out_dc = bool(_cfg(cfg, "pearson.outputs.desi_cell", False))
    pe_out_dr = bool(_cfg(cfg, "pearson.outputs.desi_roi", True))

    pe_mode   = _cfg(cfg, "pearson.sample_sets.mode",   sp_mode)    # defaults to same as spearman
    pe_custom = _cfg(cfg, "pearson.sample_sets.custom", sp_custom)

    pe_fdr_enabled = bool(_cfg(cfg, "pearson.fdr.enabled", True))
    pe_fdr_cols    = _cfg(cfg, "pearson.fdr.p_columns", ["p"])

    # -----------------------
    # Load SpatialData
    # -----------------------
    _step("Loading SpatialData")
    sdata = SpatialData.read(zarr_path)

    desi_tables = find_desi_tables(sdata, prefer_regex=desi_table_regex)
    if not desi_tables:
        raise ValueError(f"No DESI-like tables found. Tables: {list(sdata.tables.keys())}")
    desi_table = desi_tables[0]

    grid_shapes = find_grid_shapes(sdata, grid_suffix_candidates=grid_suffix_candidates)
    if not grid_shapes:
        raise ValueError(f"No grid shapes found. Shapes: {list(sdata.shapes.keys())}")

    roi_shapes = find_roi_label_shapes(sdata, roi_suffix_candidates=roi_suffix_candidates)

    print("Detected:")
    print("  DESI table:", desi_table, "| candidates:", desi_tables)
    print("  grid shapes:", grid_shapes)
    print("  ROI shapes:", roi_shapes if roi_shapes else "(none)")
    print("  coordinate systems:", get_cs_names(sdata))
    
    print("\n--- ROI SHAPE COLUMNS ---")
    for roi in roi_shapes:
        try:
            print(roi, list(sdata.shapes[roi].columns))
        except Exception as e:
            print(roi, "ERROR:", e)
    print("-------------------------\n")
    
    

    # -----------------------
    # Build pixel DF across samples
    # -----------------------
    _step("Building pixel table from grids")
    desi_all = sdata.tables[desi_table]  # AnnData
    n_total = desi_all.n_obs

    # Determine grid order deterministically (Ctrl first, then HDM) like the notebook
    grid_names = sorted(grid_shapes)

    grid_gdfs = [sdata.shapes[g] for g in grid_names]
    grid_sizes = [gdf.shape[0] for gdf in grid_gdfs]

    print("[pixels] grid_sizes:", dict(zip(grid_names, grid_sizes)))
    if sum(grid_sizes) != n_total:
        raise ValueError(
            f"Grid sizes {grid_sizes} do not sum to DESI table rows {n_total}. "
            "Cannot map 1:1. Check grids/table."
        )

    starts = np.cumsum([0] + grid_sizes[:-1])
    pixel_dfs = []

    for grid_name, n_rows, start in zip(grid_names, grid_sizes, starts):
        sample = _strip_grid_suffix(grid_name, grid_suffix_candidates)
        row_sl = slice(int(start), int(start + n_rows))

        # Subset DESI rows for this sample (exactly like notebook)
        adata = desi_all[row_sl].copy()

        # Intensities
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        intens_df = pd.DataFrame(X, columns=adata.var_names)
        if metabolite_subset:
            keep = [c for c in metabolite_subset if c in intens_df.columns]
            intens_df = intens_df[keep]

        # x,y from raw grid centroids — no coordinate transform.
        # This matches the notebook convention: one unit per DESI pixel (0–~325 range).
        # ROI annotation uses polygon_query (CS-aware internally) and matches on obs_names,
        # so x/y do not need to be in physical space for annotation to work.
        grid_gdf_raw = sdata.shapes[grid_name]
        cent = grid_gdf_raw.geometry.centroid
        xy_df = pd.DataFrame({"x": cent.x.to_numpy(), "y": cent.y.to_numpy()}).reset_index(drop=True)

        if len(xy_df) != len(intens_df):
            raise ValueError(f"{sample}: grid xy rows {len(xy_df)} != intens rows {len(intens_df)}")

        df = pd.concat([xy_df, intens_df], axis=1)
        df["sample"] = sample
        df["obs_name"] = adata.obs_names.to_numpy()          # <-- this is your real pixel id within DESI table
        df["pixel_id"] = df["sample"].astype(str) + "__" + df["obs_name"].astype(str)
        df.insert(0, "Pixel_num", np.arange(len(df), dtype=int))

        df = df.set_index("pixel_id", drop=True)
        pixel_dfs.append(df)

    pixels = pd.concat(pixel_dfs, axis=0, ignore_index=False)
    pixels = pixels.sort_values(["sample", "y", "x"]).copy()

    print(
        f"[pixels] rows={len(pixels):,} "
        f"unique_pixel_id(index)={pixels.index.nunique():,} "
        f"unique_obs_name={pixels['obs_name'].nunique():,}"
    )
    print("[pixels] rows per sample:\n", pixels["sample"].value_counts())

    # Samples list — defined here so it's always available regardless of which branches run
    samples = sorted(pixels["sample"].unique().tolist())

    # Validate halo radii against actual coordinate spacing (fails fast with clear message)
    if roi_halo_enabled:
        validate_halo_radius(
            pixels, radius=roi_halo_radius,
            physical_pixel_size_um=physical_pixel_size_um,
            label="roi_halo",
        )
    if cell_halo_enabled:
        validate_halo_radius(
            pixels, radius=cell_halo_radius,
            physical_pixel_size_um=physical_pixel_size_um,
            label="cell_halo",
        )

    # ROI join
    _step("Annotating pixels with ROI (polygon_query)")
    roi_annotation_present = False
    if join_roi and roi_shapes:
        pixels = annotate_pixels_with_roi_strtree(
            sdata=sdata,
            pixels=pixels,
            roi_shapes=roi_shapes,
            roi_label_col=str(roi_label_col or "name"),
            output_col="ROI_annotation",
            chunk_size=200_000,
            desi_table_name=desi_table,
            roi_subset=roi_subset,
        )
        pixels = make_roi_binary_columns(pixels, roi_col="ROI_annotation", prefix=roi_bin_prefix,
                                          group_regex=roi_group_regex)
        roi_annotation_present = "ROI_annotation" in pixels.columns

        if roi_halo_enabled:
            _bin_cols = [c for c in pixels.columns if c.startswith(str(roi_bin_prefix))]
            print(f"[roi_halo] adding distance-decay columns for: {_bin_cols}  "
                  f"radius={roi_halo_radius} decay={roi_halo_decay!r} suffix={roi_halo_suffix!r}")
            pixels = add_roi_distance_columns(
                pixels,
                roi_bin_cols=_bin_cols,
                radius=roi_halo_radius,
                decay=roi_halo_decay,
                suffix=roi_halo_suffix,
            )
        
        # QC: ROI overlays for each sample
        for i, sample in enumerate(samples, start=1):
            out_png = os.path.join(qc_dir, f"qc_roi__{sample}.png")
            print(f"[qc] ({i}/{len(samples)}) writing {out_png}")
            qc_plot_roi_overlay(sdata, pixels.reset_index(), sample, roi_shapes, out_png, max_points=30000)

        if write_pixel_table:
            _pixel_table_pending = True   # deferred — written after cell annotation so CELL__ cols are included
    else:
        _step("Skipping ROI annotation (disabled or no ROI shapes)")

    
    
    
    # ------------------------------------------------------------
    # Cell radius assignment: build pixel-level CELL__* features
    # ------------------------------------------------------------
    cell_features: List[str] = []  # will be overwritten by radius features if enabled
    

    if cell_enabled and cell_radius_enabled:
        _step("Annotating pixels with cells (coord-aware radius assignment)")
        # Notebook-derived coord-aware mode: transform cell shapes -> per-sample CS, align type/area from tables, KDTree containment.
        # Configure via YAML:
        #   cell.mode: "spatialdata"
        #   cell.shapes.name_regex: "..."  (optional)
        if cell_radius_mode.lower() in ("spatialdata", "shapes", "shapes+tables", "coordaware"):
            cell_shape_names = find_shapes_by_regex(sdata, cell_shapes_name_regex)

            if not cell_shape_names:
                print(f"[cell] No cell shapes found matching regex '{cell_shapes_name_regex}'. Skipping.")
            else:
                # Find which table/column holds cell type and area (search all tables like notebook)
                t_ct, ct_col = _find_table_with_cols(sdata, cell_type_cands)
                t_ar, area_col = _find_table_with_cols(sdata, cell_area_cands)
                if t_ct is None or t_ar is None:
                    print(f"[cell] Could not find required cell columns. type_table={t_ct}/{ct_col}, area_table={t_ar}/{area_col}. Skipping.")
                else:
                    print(f"[cell] cell types from: {t_ct}/{ct_col}")
                    print(f"[cell] cell area  from: {t_ar}/{area_col}")

                    # Per-sample mapping (so each sample uses its own coordinate system)
                    all_pixel_cell = []
                    all_cell_meta = []
                    all_crop_ids = []   # pixel_ids surviving the crop (if crop_to_cell_extent)

                    samples_here = sorted(pixels["sample"].unique().tolist())
                    for si, sample in enumerate(samples_here, start=1):
                        cell_shape = next((k for k in cell_shape_names if k.startswith(sample)), cell_shape_names[0])
                        print(f"[cell] ({si}/{len(samples_here)}) sample={sample} cell_shape={cell_shape}")

                        # Build pixel table for cell matching — may need transformed coords
                        # if cells are in a different coordinate space than the raw DESI grid.
                        if cell_coord_system is not None:
                            grid_name_for_sample = next(k for k in sdata.shapes if "grid" in k.lower() and sample in k)
                            grid_gdf_t = sd.transform(
                                sdata.shapes[grid_name_for_sample],
                                to_coordinate_system=cell_coord_system
                            )
                            cent = grid_gdf_t.geometry.centroid
                            obs_names_grid = sdata.shapes[grid_name_for_sample].index.astype(str)  # ensure str
                            xy_lookup = pd.DataFrame({
                                "obs_name": obs_names_grid,
                                "x_t": cent.x.to_numpy(),
                                "y_t": cent.y.to_numpy()
                            })
                            pixels_for_cells = pixels.copy()
                            pixels_for_cells["obs_name"] = pixels_for_cells["obs_name"].astype(str)  # ensure str
                            merged = pixels_for_cells.reset_index().merge(xy_lookup, on="obs_name", how="left")
                            pixels_for_cells["x"] = merged["x_t"].to_numpy()
                            pixels_for_cells["y"] = merged["y_t"].to_numpy()
                            print(f"[cell] coord_system='{cell_coord_system}': pixel x-range after transform = "
                                  f"{pixels_for_cells['x'].max() - pixels_for_cells['x'].min():.1f}")
                        else:
                            pixels_for_cells = pixels.copy()

                        if crop_to_cell_extent:
                            # Use transformed coords for bounds check if applicable
                            cell_gdf_cs = sd.transform(sdata.shapes[cell_shape], to_coordinate_system=sample)
                            minx, miny, maxx, maxy = cell_gdf_cs.total_bounds

                            g = pixels_for_cells.loc[pixels_for_cells["sample"] == sample]
                            before = len(g)
                            g = g[(g["x"] >= minx) & (g["x"] <= maxx) & (g["y"] >= miny) & (g["y"] <= maxy)]
                            after = len(g)

                            print(
                                f"[cell] crop-to-cell-extent sample={sample}: {before:,} -> {after:,} "
                                f"bounds=({minx:.2f},{miny:.2f})-({maxx:.2f},{maxy:.2f})"
                            )
                            pixels_for_cells = g.copy()
                            # Record surviving pixel_ids for permanent crop of main pixels table
                            all_crop_ids.append(pixels_for_cells.index.values)

                        pixel_cell, cell_meta = build_pixel_to_cell_radius_map_spatialdata(
                            sdata=sdata,
                            pixels_df=pixels_for_cells,          # <-- CHANGED
                            sample=sample,
                            cell_shapes_name=cell_shape,
                            target_coordinate_system=sample,
                            cell_type_table_name=t_ct,
                            cell_type_col=ct_col,
                            cell_area_table_name=t_ar,
                            cell_area_col=area_col,
                            rm_cell_types=cell_rm_types,
                            cell_radius_scalar=cell_radius_scalar,
                            progress_every_pixels=5000,
                        )
                        all_pixel_cell.append(pixel_cell)
                        all_cell_meta.append(cell_meta)

                    pixel_cell = pd.concat(all_pixel_cell, axis=0, ignore_index=True) if all_pixel_cell else pd.DataFrame(columns=["pixel_id","cell_idx","sample"])
                    cell_meta = pd.concat(all_cell_meta, axis=0, ignore_index=True) if all_cell_meta else pd.DataFrame()

                    # Permanently crop the main pixels table to the cell extent.
                    # all_crop_ids is only populated when crop_to_cell_extent=true.
                    if crop_to_cell_extent and all_crop_ids:
                        keep_ids = np.concatenate(all_crop_ids)
                        pixels_before_crop = pixels.copy()   # snapshot for QC plot only
                        pixels = pixels[pixels.index.isin(keep_ids)].copy()
                        print(f"[cell] permanent crop: {len(pixels_before_crop):,} -> {len(pixels):,} pixels "
                              f"(all downstream outputs use cropped table)")
                    else:
                        pixels_before_crop = None

                    # QC plot: two-panel before/after if cropped, single panel otherwise
                    _step("QC plots: cell overlays")
                    for i, sample in enumerate(samples_here, start=1):
                        cell_shape_name = next((k for k in cell_shape_names if k.startswith(sample)), cell_shape_names[0])
                        out_png = os.path.join(qc_dir, f"qc_cells__{sample}.png")
                        print(f"[qc] ({i}/{len(samples_here)}) writing {out_png}")
                        qc_plot_cell_overlay(
                            sdata, pixels.reset_index(), sample, cell_shape_name, out_png,
                            max_points=30000, n_cells=2000,
                            roi_shapes=roi_shapes if roi_shapes else None,
                            pixels_before_crop=pixels_before_crop.reset_index() if pixels_before_crop is not None else None,
                        )

                    if cell_halo_enabled and not cell_meta.empty:
                        validate_cell_coordinate_space(pixels_for_cells, cell_meta)

                    if write_cell_meta_csv and (not cell_meta.empty):
                        cell_meta.to_csv(os.path.join(out_dir, "cell_meta.csv"), index=False)
                    if write_pixel_cell_map_csv and (not pixel_cell.empty):
                        pixel_cell.to_csv(os.path.join(out_dir, "pixel_to_cell_map.csv"), index=False)

                    # Build pixel-level CELL__* features (uses pixel_id)
                    pixels, cell_features = pixel_celltype_features(
                        pixels_df=pixels,
                        pixel_cell=pixel_cell[["pixel_id", "cell_idx"]],
                        cell_meta=cell_meta[["cell_idx", "cell_type"]],
                        mode=cell_feat_mode,
                        prefix=cell_prefix,
                    )

                    if cell_halo_enabled and cell_features:
                        print(f"[cell_halo] adding halo columns for {len(cell_features)} cell types  "
                              f"halo_radius={cell_halo_radius} decay={cell_halo_decay!r}")
                        pixels = add_cell_halo_columns(
                            pixels, cell_meta, cell_features,
                            halo_radius=cell_halo_radius,
                            decay=cell_halo_decay,
                            suffix=cell_halo_suffix,
                        )
                        cell_features = [c for c in pixels.columns if c.startswith(cell_prefix)]

                    print(f"[cell] Coord-aware radius assignment: added {len(cell_features)} pixel-level cell-type columns.")
                    for cf in cell_features:
                        n_nonzero = (pixels[cf] > 0).sum()
                        if n_nonzero == 0:
                            print(f"[cell] WARNING: {cf} has 0 pixels assigned — will be dropped from correlations (zero variance)")
                        else:
                            print(f"[cell]   {cf}: {n_nonzero:,}/{len(pixels):,} pixels > 0")
        else:
            # Legacy table mode (assumes x/y already in the same space as pixels)
            if cell_radius_mode != "table":
                print("[cell] radius_assignment expects mode='spatialdata' or mode='table'. Skipping.")
            else:
                cell_table_name = find_table_by_regex(sdata, cell_table_name_regex)
                if cell_table_name is None:
                    print(f"[cell] No cell table found matching regex '{cell_table_name_regex}'. Skipping radius assignment.")
                else:
                    ad = sdata.tables[cell_table_name]
                    cell_df = ad.obs.copy()

                    x_col = pick_first_existing_col_df(cell_df, cell_x_cands)
                    y_col = pick_first_existing_col_df(cell_df, cell_y_cands)
                    area_col = pick_first_existing_col_df(cell_df, cell_area_cands)
                    type_col = pick_first_existing_col_df(cell_df, cell_type_cands)

                    if None in (x_col, y_col, area_col, type_col):
                        print(f"[cell] Missing cols in {cell_table_name}. x={x_col}, y={y_col}, area={area_col}, type={type_col}")
                    else:
                        pixel_cell, cell_meta = build_pixel_to_cell_radius_map_from_table(
                            pixels_df=pixels,
                            cell_df=cell_df,
                            cell_x_col=x_col,
                            cell_y_col=y_col,
                            cell_area_col=area_col,
                            cell_type_col=type_col,
                            rm_cell_types=cell_rm_types,
                            cell_radius_scalar=cell_radius_scalar,
                        )

                        if write_cell_meta_csv:
                            cell_meta.to_csv(os.path.join(out_dir, "cell_meta.csv"), index=False)
                        if write_pixel_cell_map_csv:
                            pixel_cell.to_csv(os.path.join(out_dir, "pixel_to_cell_map.csv"), index=False)

                        pixels, cell_features = pixel_celltype_features(
                            pixels_df=pixels,
                            pixel_cell=pixel_cell,
                            cell_meta=cell_meta,
                            mode=cell_feat_mode,
                            prefix=cell_prefix,
                        )

                        if cell_halo_enabled and cell_features:
                            print(f"[cell_halo] adding halo columns for {len(cell_features)} cell types  "
                                  f"halo_radius={cell_halo_radius} decay={cell_halo_decay!r}")
                            pixels = add_cell_halo_columns(
                                pixels, cell_meta, cell_features,
                                halo_radius=cell_halo_radius,
                                decay=cell_halo_decay,
                                suffix=cell_halo_suffix,
                            )
                            cell_features = [c for c in pixels.columns if c.startswith(cell_prefix)]

                        print(f"[cell] Legacy radius assignment: added {len(cell_features)} pixel-level cell-type columns.")


    # ----------------------------------------------------------------
    # Deferred pixel table write — now includes CELL__* and halo cols
    # ----------------------------------------------------------------
    if locals().get("_pixel_table_pending", False):
        decay_cols = [c for c in pixels.columns
                      if c.startswith(str(roi_bin_prefix)) and c.endswith(str(roi_halo_suffix))]
        if decay_cols:
            _pfx_len = len(str(roi_bin_prefix))
            _sfx_len = len(str(roi_halo_suffix))
            _decay_label = {c: c[_pfx_len:-_sfx_len] for c in decay_cols}
            _decay_vals  = pixels[decay_cols]
            _max_decay   = _decay_vals.max(axis=1)
            _best_col    = _decay_vals.idxmax(axis=1)
            annotation_decay = _best_col.map(_decay_label)
            annotation_decay[_max_decay == 0] = None
        else:
            annotation_decay = pd.Series([None] * len(pixels), index=pixels.index)

        desi_meta_cols = {"x", "y", "sample", "obs_name", "Pixel_num", "ROI_annotation"}
        metabolite_cols_here = [
            c for c in pixels.columns
            if c not in desi_meta_cols
            and not (c.startswith(str(roi_bin_prefix)) and not c.endswith(str(roi_halo_suffix)))
            and pd.api.types.is_numeric_dtype(pixels[c])
        ]

        combined = pixels[["x", "y"] + metabolite_cols_here].copy()
        combined["annotation"]       = pixels["ROI_annotation"]
        combined["annotation_decay"] = annotation_decay.values
        combined["sample"]           = pixels["sample"]
        combined["pixel_id"]         = pixels["obs_name"]
        combined = combined.reset_index(drop=True)
        combined.insert(0, "Pixel_num", combined.index)
        combined.to_csv(os.path.join(out_dir, "pixels_with_annotation_combined.csv"), index=False)
        print("[roi] wrote pixels_with_annotation_combined.csv")

        out_long = pixels.copy()
        out_long = out_long.rename(columns={"ROI_annotation": "annotation"})
        out_long["annotation_decay"] = annotation_decay.values
        out_long = out_long.drop(columns=["Pixel_num"], errors="ignore")
        out_long.insert(0, "Pixel_num", range(1, len(out_long) + 1))
        out_long.to_csv(os.path.join(out_dir, "pixels_with_annotation_long.csv"), index=True)
        print("[roi] wrote pixels_with_annotation_long.csv")

    # ----------------------------------------------------------------
    # Hotspot neighbourhood analysis
    # ----------------------------------------------------------------
    hs_cfg = cfg.get("hotspot_neighbourhood", {}) or {}
    hs_enabled = bool(hs_cfg.get("enabled", False))
    if hs_enabled:
        _step("Hotspot neighbourhood analysis")
        from scipy.spatial import KDTree

        hs_radius_default   = float(hs_cfg.get("radius", 15))
        hs_quantile_default = float(hs_cfg.get("hotspot_threshold_quantile", 0.90))
        hs_anchors          = hs_cfg.get("anchors", []) or []

        hs_out_dir = os.path.join(out_dir, "hotspot_neighbourhood")
        os.makedirs(hs_out_dir, exist_ok=True)

        # Pre-compute column lists used for neighbourhood summaries
        _base_excl = {"x", "y", "sample", "ROI_annotation", "Pixel_num", "obs_name"}
        _roi_bin   = [c for c in pixels.columns if c.startswith(str(roi_bin_prefix))
                      and not c.endswith(str(roi_halo_suffix))]
        _hs_desi_cols = [c for c in pixels.columns
                         if c not in _base_excl and c not in set(_roi_bin)
                         and not c.startswith(cell_prefix)
                         and pd.api.types.is_numeric_dtype(pixels[c])]
        _hs_cell_cols = [c for c in pixels.columns if c.startswith(cell_prefix)
                         and not c.endswith("_halo")]

        coords = pixels[["x", "y"]].to_numpy()
        tree   = KDTree(coords)

        for anchor_cfg in hs_anchors:
            anchor_type = anchor_cfg.get("anchor_type", "")
            anchor_name = anchor_cfg.get("anchor_name", "")
            radius      = float(anchor_cfg.get("radius", hs_radius_default))
            quantile    = float(anchor_cfg.get("hotspot_threshold_quantile", hs_quantile_default))

            # --- Identify hotspot mask ---
            if anchor_type == "desi_obs":
                if anchor_name not in pixels.columns:
                    print(f"[hotspot] WARNING: column '{anchor_name}' not found — skipping")
                    continue
                col = pixels[anchor_name]
                thresh = col.quantile(quantile)
                hotspot_mask = (col >= thresh).values
                anchor_vals  = col.values

            elif anchor_type == "cell_type":
                col_name = f"{cell_prefix}{anchor_name}"
                if col_name not in pixels.columns:
                    print(f"[hotspot] WARNING: column '{col_name}' not found — skipping")
                    continue
                hotspot_mask = (pixels[col_name] > 0).values
                anchor_vals  = pixels[col_name].values

            elif anchor_type == "roi":
                roi_col = f"{roi_bin_prefix}{anchor_name}"
                if roi_col in pixels.columns:
                    hotspot_mask = (pixels[roi_col] > 0).values
                elif "ROI_annotation" in pixels.columns:
                    hotspot_mask = (pixels["ROI_annotation"] == anchor_name).values
                else:
                    print(f"[hotspot] WARNING: ROI '{anchor_name}' not found — skipping")
                    continue
                anchor_vals = hotspot_mask.astype(float)

            else:
                print(f"[hotspot] WARNING: unknown anchor_type '{anchor_type}' — skipping")
                continue

            n_hotspots = hotspot_mask.sum()
            print(f"[hotspot] anchor={anchor_name!r} type={anchor_type} "
                  f"radius={radius} hotspots={n_hotspots:,}")
            if n_hotspots == 0:
                print(f"[hotspot]   → no hotspot pixels, skipping")
                continue

            hotspot_indices = np.where(hotspot_mask)[0]
            rows = []
            for hi in hotspot_indices:
                nb_idx = tree.query_ball_point(coords[hi], r=radius)
                nb_idx = [j for j in nb_idx if j != hi]
                nb = pixels.iloc[nb_idx] if nb_idx else pixels.iloc[[]]
                n_nb = len(nb)

                row = {
                    "anchor_type":    anchor_type,
                    "anchor_name":    anchor_name,
                    "anchor_value":   float(anchor_vals[hi]),
                    "sample":         pixels.iloc[hi]["sample"],
                    "x":              float(coords[hi, 0]),
                    "y":              float(coords[hi, 1]),
                    "n_neighbours":   n_nb,
                    "roi_annotation": pixels.iloc[hi].get("ROI_annotation", None),
                }
                for dc in _hs_desi_cols:
                    row[f"mean_{dc}"] = float(nb[dc].mean()) if n_nb > 0 else float("nan")
                for cc in _hs_cell_cols:
                    row[f"frac_{cc}"] = float((nb[cc] > 0).mean()) if n_nb > 0 else float("nan")
                rows.append(row)

            safe_name = re.sub(r"[^\w\-]", "_", anchor_name)
            out_csv = os.path.join(hs_out_dir, f"{safe_name}.csv")
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[hotspot]   → wrote {len(rows):,} rows to {out_csv}")

    # DESI columns — exclude all non-metabolite columns
    base_cols = {"x", "y", "sample", "ROI_annotation", "Pixel_num", "obs_name"}
    roi_bin_cols = [c for c in pixels.columns if c.startswith(str(roi_bin_prefix))]
    exclude = base_cols | set(roi_bin_cols) | set(cell_features)
    desi_cols = [c for c in pixels.columns if c not in exclude and pd.api.types.is_numeric_dtype(pixels[c])]
    if not desi_cols:
        raise ValueError("No DESI numeric columns detected.")

    # Samples / sample sets — samples list already defined above
    sp_sample_sets = make_sample_sets(samples, mode=sp_mode, custom=sp_custom)
    pe_sample_sets = make_sample_sets(samples, mode=pe_mode, custom=pe_custom)

    # DESI-DESI pairs
    desi_desi_pairs = make_upper_triangle_pairs(desi_cols)
    if max_desi_desi_pairs is not None and len(desi_desi_pairs) > int(max_desi_desi_pairs):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(desi_desi_pairs), size=int(max_desi_desi_pairs), replace=False)
        desi_desi_pairs = [desi_desi_pairs[i] for i in idx]

    # Pair lists
    pairs_desi_cell = make_pairs(desi_cols, cell_features) if cell_features else []
    pairs_desi_roi = make_pairs(desi_cols, roi_bin_cols) if roi_bin_cols else []

    # Choose p/z sources for combining
    # Prefer p_sim/z_sim if permutations ran OR if p_norm is absent/all-NaN
    prefer_sim = perm_enabled and use_p_sim_for_stats
    p_source = "p_sim" if prefer_sim else "p_norm"
    z_source = "z_sim" if prefer_sim else "z_norm"

    _step("Running Moran")

    # -----------------------
    # Run MORAN outputs (toggled)
    # -----------------------
    if moran_enabled:
        if moran_out_uni:
            uni_ps = moran_univariate_per_sample(
                pixels, desi_cols,
                wtype=wtype,
                threshold=threshold,
                threshold_multiplier=thr_mult,
                knn_k=knn_k,
                binary=binary,
                permutations=perm_n,
                cache_dir=weights_cache_dir,
            )
            if moran_fdr_enabled:
                uni_ps = add_fdr_columns(uni_ps, moran_fdr_cols, prefix="q_")
            uni_ps.to_csv(os.path.join(out_dir, "moran_univariate_per_sample.csv"), index=False)

            if combine_enabled:
                uni_comb = combine_moran_effects(
                    uni_ps, group_cols=["x_var"],
                    p_source=p_source, z_source=z_source,
                    stouffer_weight=stouffer_weight,
                )
                if moran_fdr_enabled:
                    uni_comb = add_fdr_columns(uni_comb, moran_fdr_cols, prefix="q_")
                uni_comb.to_csv(os.path.join(out_dir, "moran_univariate_combined.csv"), index=False)

        if moran_out_dd:
            bv_dd_ps = moran_bivariate_per_sample(
                pixels, desi_desi_pairs,
                wtype=wtype,
                threshold=threshold,
                threshold_multiplier=thr_mult,
                knn_k=knn_k,
                binary=binary,
                permutations=perm_n,
                cache_dir=weights_cache_dir,
            )
            if moran_fdr_enabled:
                bv_dd_ps = add_fdr_columns(bv_dd_ps, moran_fdr_cols, prefix="q_")
            bv_dd_ps.to_csv(os.path.join(out_dir, "moran_bivariate_desi_vs_desi_per_sample.csv"), index=False)

            if combine_enabled:
                bv_dd_comb = combine_moran_effects(
                    bv_dd_ps, group_cols=["x_var", "y_var"],
                    p_source=p_source, z_source=z_source,
                    stouffer_weight=stouffer_weight,
                )
                if moran_fdr_enabled:
                    bv_dd_comb = add_fdr_columns(bv_dd_comb, moran_fdr_cols, prefix="q_")
                bv_dd_comb.to_csv(os.path.join(out_dir, "moran_bivariate_desi_vs_desi_combined.csv"), index=False)

        if moran_out_dc and pairs_desi_cell:
            bv_dc_ps = moran_bivariate_per_sample(
                pixels, pairs_desi_cell,
                wtype=wtype,
                threshold=threshold,
                threshold_multiplier=thr_mult,
                knn_k=knn_k,
                binary=binary,
                permutations=perm_n,
                cache_dir=weights_cache_dir,
            )
            if moran_fdr_enabled:
                bv_dc_ps = add_fdr_columns(bv_dc_ps, moran_fdr_cols, prefix="q_")
            bv_dc_ps.to_csv(os.path.join(out_dir, "moran_bivariate_desi_vs_cell_per_sample.csv"), index=False)

            if combine_enabled:
                bv_dc_comb = combine_moran_effects(
                    bv_dc_ps, group_cols=["x_var", "y_var"],
                    p_source=p_source, z_source=z_source,
                    stouffer_weight=stouffer_weight,
                )
                if moran_fdr_enabled:
                    bv_dc_comb = add_fdr_columns(bv_dc_comb, moran_fdr_cols, prefix="q_")
                bv_dc_comb.to_csv(os.path.join(out_dir, "moran_bivariate_desi_vs_cell_combined.csv"), index=False)

        if moran_out_dr and pairs_desi_roi:
            bv_dr_ps = moran_bivariate_per_sample(
                pixels, pairs_desi_roi,
                wtype=wtype,
                threshold=threshold,
                threshold_multiplier=thr_mult,
                knn_k=knn_k,
                binary=binary,
                permutations=perm_n,
                cache_dir=weights_cache_dir,
            )
            if moran_fdr_enabled:
                bv_dr_ps = add_fdr_columns(bv_dr_ps, moran_fdr_cols, prefix="q_")
            bv_dr_ps.to_csv(os.path.join(out_dir, "moran_bivariate_desi_vs_roi_per_sample.csv"), index=False)

            if combine_enabled:
                bv_dr_comb = combine_moran_effects(
                    bv_dr_ps, group_cols=["x_var", "y_var"],
                    p_source=p_source, z_source=z_source,
                    stouffer_weight=stouffer_weight,
                )
                if moran_fdr_enabled:
                    bv_dr_comb = add_fdr_columns(bv_dr_comb, moran_fdr_cols, prefix="q_")
                bv_dr_comb.to_csv(os.path.join(out_dir, "moran_bivariate_desi_vs_roi_combined.csv"), index=False)

        # -----------------------
        # Local Moran outputs (LISA)
        # -----------------------
        if local_enabled:
            # 1) Local univariate summary per metabolite
            loc_uni = local_moran_univariate_summary_per_sample(
                pixels,
                value_cols=desi_cols,
                wtype=wtype,
                threshold=threshold,
                threshold_multiplier=thr_mult,
                knn_k=knn_k,
                binary=binary,
                permutations=perm_n,
                alpha=local_alpha,
                fdr_on_local_p=local_fdr_on_p,
                p_source=local_p_source,
                cache_dir=weights_cache_dir,
            )
            loc_uni.to_csv(os.path.join(out_dir, "local_moran_univariate_summary.csv"), index=False)

            # 2) Local bivariate summary: DESI vs DESI
            if local_bv_dd_enabled and desi_desi_pairs:
                local_dd_pairs = _resolve_local_pairs(desi_desi_pairs, local_bv_dd_subset, "desi_vs_desi")
                if local_dd_pairs:
                    loc_bv_dd_dir = os.path.join(out_dir, "local_bv", "met_met")
                    os.makedirs(loc_bv_dd_dir, exist_ok=True)
                    print(f"[local_bv] DESI vs DESI: {len(local_dd_pairs)} pairs")
                    loc_bv_dd = local_moran_bivariate_summary_per_sample(
                        pixels,
                        pairs=local_dd_pairs,
                        wtype=wtype,
                        threshold=threshold,
                        threshold_multiplier=thr_mult,
                        knn_k=knn_k,
                        binary=binary,
                        permutations=perm_n,
                        alpha=local_alpha,
                        fdr_on_local_p=local_fdr_on_p,
                        pixel_out_dir=loc_bv_dd_dir if local_write_pixel_maps else None,
                        pixel_index_col="obs_name",
                        cache_dir=weights_cache_dir,
                    )
                    loc_bv_dd.to_csv(os.path.join(loc_bv_dd_dir, "local_moran_bivariate_desi_vs_desi_summary.csv"), index=False)

            # 3) Local bivariate summary: DESI vs CELL__*
            if local_bv_dc_enabled and pairs_desi_cell:
                local_dc_pairs = _resolve_local_pairs(pairs_desi_cell, local_bv_dc_subset, "desi_vs_cell")
                if local_dc_pairs:
                    loc_bv_dc_dir = os.path.join(out_dir, "local_bv", "met_cell")
                    os.makedirs(loc_bv_dc_dir, exist_ok=True)
                    print(f"[local_bv] DESI vs CELL: {len(local_dc_pairs)} pairs")
                    loc_bv_cell = local_moran_bivariate_summary_per_sample(
                        pixels,
                        pairs=local_dc_pairs,
                        wtype=wtype,
                        threshold=threshold,
                        threshold_multiplier=thr_mult,
                        knn_k=knn_k,
                        binary=binary,
                        permutations=perm_n,
                        alpha=local_alpha,
                        fdr_on_local_p=local_fdr_on_p,
                        pixel_out_dir=loc_bv_dc_dir if local_write_pixel_maps else None,
                        pixel_index_col="obs_name",
                        cache_dir=weights_cache_dir,
                    )
                    loc_bv_cell.to_csv(os.path.join(loc_bv_dc_dir, "local_moran_bivariate_desi_vs_cell_summary.csv"), index=False)

            # 4) Local bivariate summary: DESI vs ROI__*
            if local_bv_dr_enabled and pairs_desi_roi:
                local_dr_pairs = _resolve_local_pairs(pairs_desi_roi, local_bv_dr_subset, "desi_vs_roi")
                if local_dr_pairs:
                    loc_bv_dr_dir = os.path.join(out_dir, "local_bv", "met_roi")
                    os.makedirs(loc_bv_dr_dir, exist_ok=True)
                    print(f"[local_bv] DESI vs ROI: {len(local_dr_pairs)} pairs")
                    loc_bv_roi = local_moran_bivariate_summary_per_sample(
                        pixels,
                        pairs=local_dr_pairs,
                        wtype=wtype,
                        threshold=threshold,
                        threshold_multiplier=thr_mult,
                        knn_k=knn_k,
                        binary=binary,
                        permutations=perm_n,
                        alpha=local_alpha,
                        fdr_on_local_p=local_fdr_on_p,
                        pixel_out_dir=loc_bv_dr_dir if local_write_pixel_maps else None,
                        pixel_index_col="obs_name",
                        cache_dir=weights_cache_dir,
                    )
                    loc_bv_roi.to_csv(os.path.join(loc_bv_dr_dir, "local_moran_bivariate_desi_vs_roi_summary.csv"), index=False)

    _step("Running Spearman")

    # -----------------------
    # Run SPEARMAN outputs (toggled)
    # -----------------------
    if spearman_enabled:
        if sp_out_dd:
            sp = spearman_pairs(pixels, desi_desi_pairs, sample_sets=sp_sample_sets)
            if sp_fdr_enabled:
                sp = add_fdr_columns(sp, sp_fdr_cols, prefix="q_")
            sp.to_csv(os.path.join(out_dir, "spearman_desi_vs_desi.csv"), index=False)

        if sp_out_dc and pairs_desi_cell:
            sp = spearman_pairs(pixels, pairs_desi_cell, sample_sets=sp_sample_sets)
            if sp_fdr_enabled:
                sp = add_fdr_columns(sp, sp_fdr_cols, prefix="q_")
            sp.to_csv(os.path.join(out_dir, "spearman_desi_vs_cell.csv"), index=False)

        if sp_out_dr and pairs_desi_roi:
            sp = spearman_pairs(pixels, pairs_desi_roi, sample_sets=sp_sample_sets)
            if sp_fdr_enabled:
                sp = add_fdr_columns(sp, sp_fdr_cols, prefix="q_")
            sp.to_csv(os.path.join(out_dir, "spearman_desi_vs_roi.csv"), index=False)

    if pearson_enabled:
        _step("Running Pearson")
        if pe_out_dd:
            pe = pearson_pairs(pixels, desi_desi_pairs, sample_sets=pe_sample_sets)
            if pe_fdr_enabled:
                pe = add_fdr_columns(pe, pe_fdr_cols, prefix="q_")
            pe.to_csv(os.path.join(out_dir, "pearson_desi_vs_desi.csv"), index=False)

        if pe_out_dc and pairs_desi_cell:
            pe = pearson_pairs(pixels, pairs_desi_cell, sample_sets=pe_sample_sets)
            if pe_fdr_enabled:
                pe = add_fdr_columns(pe, pe_fdr_cols, prefix="q_")
            pe.to_csv(os.path.join(out_dir, "pearson_desi_vs_cell.csv"), index=False)

        if pe_out_dr and pairs_desi_roi:
            pe = pearson_pairs(pixels, pairs_desi_roi, sample_sets=pe_sample_sets)
            if pe_fdr_enabled:
                pe = add_fdr_columns(pe, pe_fdr_cols, prefix="q_")
            pe.to_csv(os.path.join(out_dir, "pearson_desi_vs_roi.csv"), index=False)

    _step("Writing metadata")

    # -----------------------
    # Metadata
    # -----------------------
    meta = dict(
        zarr_path=zarr_path,
        desi_table=desi_table,
        n_pixels=int(pixels.shape[0]),
        samples=samples,
        n_desi=len(desi_cols),
        n_cell=len(cell_features),
        n_roi_bins=len(roi_bin_cols),
        permutations=perm_n,
        p_source=p_source,
        z_source=z_source,
        spearman_sample_sets=["+".join(ss) for ss in sp_sample_sets],
        threshold=threshold,
        binary=binary,
        max_desi_desi_pairs=max_desi_desi_pairs,
    )
    pd.Series(meta).to_csv(os.path.join(out_dir, "run_metadata.csv"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_desi_from_config.py config.yaml")
    run_from_yaml(sys.argv[1])
# desi_spatial_helpers.py
from __future__ import annotations

import re
import itertools
import hashlib
import pickle
import logging
import shapely
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, chi2, norm

import matplotlib.pyplot as plt

try:
    from statsmodels.stats.multitest import multipletests
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

from spatialdata import SpatialData
import time
import spatialdata as sd

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from libpysal.weights import DistanceBand
    from esda.moran import Moran, Moran_BV
    from esda.moran import Moran_Local
    from esda.moran import Moran_Local_BV
except Exception:
    DistanceBand = None
    Moran = None
    Moran_BV = None
    Moran_Local = None
    Moran_Local_BV = None

from libpysal.weights import KNN
from shapely.geometry import Point, Polygon, MultiPolygon
from sklearn.neighbors import KDTree


# ----------------------------
# Detection
# ----------------------------

def get_cs_names(sdata: SpatialData) -> List[str]:
    cs = sdata.coordinate_systems
    return list(cs.keys()) if hasattr(cs, "keys") else list(cs)

def _match_any(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, name, flags=re.IGNORECASE) for p in patterns)

def find_desi_tables(sdata: SpatialData, prefer_regex: Sequence[str]) -> List[str]:
    keys = list(getattr(sdata, "tables", {}).keys())
    ranked = [k for k in keys if _match_any(k, prefer_regex)]
    ranked = sorted(ranked, key=lambda x: (0 if re.search(r"_obs$", x, re.I) else 1, len(x), x))
    return ranked

def find_grid_shapes(sdata: SpatialData, grid_suffix_candidates: Sequence[str]) -> List[str]:
    keys = list(getattr(sdata, "shapes", {}).keys())
    grids = [k for k in keys if any(k.endswith(suf) for suf in grid_suffix_candidates)]
    return sorted(grids)

def find_roi_label_shapes(sdata: SpatialData, roi_suffix_candidates: Sequence[str]) -> List[str]:
    keys = list(getattr(sdata, "shapes", {}).keys())
    rois = []
    for k in keys:
        if any(k.endswith(suf) for suf in roi_suffix_candidates) or _match_any(k, (r"anat", r"roi", r"label")):
            rois.append(k)
    return sorted(set(rois))

def find_cell_tables(sdata: SpatialData, cell_table_regex: Sequence[str]) -> List[str]:
    keys = list(getattr(sdata, "tables", {}).keys())
    cands = [k for k in keys if _match_any(k, cell_table_regex)]
    return sorted(cands)

def choose_sample_from_shape_name(shape_name: str, cs_names: Sequence[str]) -> Optional[str]:
    for cs in cs_names:
        if shape_name.startswith(cs):
            return cs
    return None


# ----------------------------
# Data extraction / ROI
# ----------------------------

def _geom_to_polygon(geom):
    # Match notebook behavior: if MultiPolygon, take largest piece; then buffer(0) to fix validity
    if isinstance(geom, MultiPolygon) or getattr(geom, "geom_type", None) == "MultiPolygon":
        geom = max(list(geom.geoms), key=lambda g: g.area)
    if not hasattr(geom, "exterior"):
        raise ValueError(f"Geometry type '{getattr(geom, 'geom_type', type(geom))}' has no exterior.")
    return Polygon(geom.exterior.coords).buffer(0)


def annotate_pixels_with_roi_strtree(
    sdata,
    pixels: pd.DataFrame,
    roi_shapes: list[str],
    roi_label_col: str = "name",
    output_col: str = "ROI_annotation",
    chunk_size: int = 200_000,  # kept for API compatibility, not used
    desi_table_name: str = "DESI_NORM_obs",
    roi_subset: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    ROI annotation using sd.polygon_query (coordinate-system aware).

    Parameters
    ----------
    sdata            : SpatialData object
    pixels           : pixel DataFrame with columns [sample, x, y] and index = pixel_id
    roi_shapes       : list of shape names to use as ROI sources
    roi_label_col    : column in the ROI GeoDataFrame that holds the region name
    output_col       : column name to write annotations into
    chunk_size       : unused, kept for backward compatibility
    desi_table_name  : name of the DESI table in sdata (needed for polygon_query lookup)
    roi_subset       : if provided, only annotate pixels within these named ROIs
                       (e.g. ["Cortex", "Hippocampus"]). Useful to speed up test runs.
    """
    roi_subset_set = set(roi_subset) if roi_subset else None
    out_parts = []
    samples = sorted(pixels["sample"].unique().tolist())
    n_samples = len(samples)

    for si, sample in enumerate(samples, start=1):
        t0 = time.time()
        g = pixels.loc[pixels["sample"] == sample].copy()
        g[output_col] = pd.NA

        roi_name = next((k for k in roi_shapes if k.startswith(sample)), roi_shapes[0])
        roi_gdf = sdata.shapes[roi_name]

        if roi_label_col not in roi_gdf.columns:
            raise ValueError(
                f"[roi] roi_label_col='{roi_label_col}' not in {roi_name} columns: {list(roi_gdf.columns)}"
            )

        n_rois = len(roi_gdf)
        if roi_subset_set:
            available = set(roi_gdf[roi_label_col].astype(str))
            missing = roi_subset_set - available
            if missing:
                print(f"[roi]   warning: roi_subset names not found in {roi_name}: {sorted(missing)}")
            print(f"[roi] ({si}/{n_samples}) sample={sample} pixels={len(g):,} rois={n_rois} "
                  f"(subset: {sorted(roi_subset_set & available)}) method=polygon_query")
        else:
            print(f"[roi] ({si}/{n_samples}) sample={sample} pixels={len(g):,} rois={n_rois} method=polygon_query")

        # Build a map: plain obs_name (str) -> composite pixel_id index used in g
        # g.index = "sample__obsname", g["obs_name"] = plain obsname
        obs_to_pixel_id = pd.Series(g.index, index=g["obs_name"].astype(str)).to_dict()

        assigned = 0
        for i in range(n_rois):
            roi_label = str(roi_gdf[roi_label_col].iloc[i])

            # Skip if a subset is requested and this ROI is not in it
            if roi_subset_set and roi_label not in roi_subset_set:
                continue
            geom = roi_gdf.geometry.iloc[i]
            roi_poly = _geom_to_polygon(geom)

            try:
                crop = sd.polygon_query(sdata, roi_poly, target_coordinate_system=sample)
            except Exception as e:
                print(f"[roi]   skipping roi='{roi_label}': polygon_query error: {e}")
                continue

            if desi_table_name not in crop.tables:
                continue

            # polygon_query returns plain obs_names — map back to composite pixel_id index
            roi_plain_ids = crop.tables[desi_table_name].obs_names.astype(str)
            roi_pixel_ids = [obs_to_pixel_id[oid] for oid in roi_plain_ids if oid in obs_to_pixel_id]
            if len(roi_pixel_ids) == 0:
                continue

            # Assign label only where not yet assigned (first-label-wins)
            unassigned_mask = g.loc[roi_pixel_ids, output_col].isna()
            targets = [pid for pid, is_na in zip(roi_pixel_ids, unassigned_mask) if is_na]
            if targets:
                g.loc[targets, output_col] = roi_label
                assigned += len(targets)

        g[output_col] = g[output_col].astype("category")
        dt = time.time() - t0
        frac = g[output_col].notna().mean() * 100.0
        print(f"[roi]   done sample={sample} assigned={assigned:,}/{len(g):,} ({frac:.1f}%) time={dt:.1f}s")

        out_parts.append(g)

    return pd.concat(out_parts, axis=0)

def densify_X(adata) -> np.ndarray:
    X = adata.X
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def grid_centroids_xy(grid_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    cent = grid_gdf.geometry.centroid
    return pd.DataFrame({"x": cent.x.to_numpy(), "y": cent.y.to_numpy()},
                        index=grid_gdf.index.astype(str))

def _guess_roi_label_col(gdf: "gpd.GeoDataFrame") -> str:
    for c in gdf.columns:
        if c.lower() == "geometry":
            continue
        if c.lower() in ("label", "labels", "annotation", "name", "region", "roi", "class"):
            return c
    for c in gdf.columns:
        if c.lower() != "geometry":
            return c
    raise ValueError("Could not guess ROI label column (GeoDataFrame has only geometry?)")

def extract_desi_pixels_for_grid(
    sdata: SpatialData,
    desi_table_name: str,
    grid_shape_name: str,
    metabolite_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if gpd is None:
        raise ImportError("geopandas required for grid operations.")

    cs_names = get_cs_names(sdata)
    sample = choose_sample_from_shape_name(grid_shape_name, cs_names) or "sample"

    grid_gdf = sdata.shapes[grid_shape_name]
    grid_ids = grid_gdf.index.astype(str)

    adata = sdata.tables[desi_table_name]

    obs_names = adata.obs_names.astype(str)
    common = np.intersect1d(obs_names, grid_ids)
    if len(common) == 0:
        possible_cols = [c for c in adata.obs.columns if re.search(r"(grid|pixel|spot|bin|id)", c, re.I)]
        found = None
        for c in possible_cols:
            vals = adata.obs[c].astype(str).to_numpy()
            inter = np.intersect1d(vals, grid_ids)
            if len(inter) > 0:
                found = c
                adata = adata[adata.obs[c].astype(str).isin(inter)].copy()
                adata.obs_names = adata.obs[c].astype(str)
                break
        if found is None:
            raise ValueError(
                f"Could not match DESI table '{desi_table_name}' rows to grid '{grid_shape_name}'."
            )
    else:
        adata = adata[obs_names.isin(common)].copy()

    X = densify_X(adata)
    var_names = adata.var_names.astype(str).to_list()

    if metabolite_cols is None:
        use_idx = np.arange(len(var_names))
        use_names = var_names
    else:
        metabolite_cols = [str(m) for m in metabolite_cols]
        idx_map = {n: i for i, n in enumerate(var_names)}
        missing = [m for m in metabolite_cols if m not in idx_map]
        if missing:
            raise ValueError(f"Requested metabolites not in '{desi_table_name}': {missing[:10]}")
        use_idx = np.array([idx_map[m] for m in metabolite_cols], dtype=int)
        use_names = metabolite_cols

    df = pd.DataFrame(X[:, use_idx], index=adata.obs_names.astype(str), columns=use_names)

    xy = grid_centroids_xy(grid_gdf.loc[grid_gdf.index.astype(str).isin(df.index)])
    df = df.join(xy, how="inner")

    df["sample"] = sample
    df.index.name = "pixel_id"
    return df

def add_roi_annotations_points_in_polygons(
    pixels_df: pd.DataFrame,
    roi_gdf: "gpd.GeoDataFrame",
    roi_label_col: Optional[str] = None,
    output_col: str = "ROI_annotation",
) -> pd.DataFrame:
    if gpd is None:
        raise ImportError("geopandas required for ROI operations.")

    if roi_label_col is None:
        roi_label_col = _guess_roi_label_col(roi_gdf)

    pts = gpd.GeoDataFrame(
        pixels_df.copy(),
        geometry=gpd.points_from_xy(pixels_df["x"], pixels_df["y"]),
        crs=getattr(roi_gdf, "crs", None),
    )
    joined = gpd.sjoin(pts, roi_gdf[[roi_label_col, "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    out[output_col] = out[roi_label_col].astype("category")
    out = out.drop(columns=[roi_label_col], errors="ignore")
    return out

def make_roi_binary_columns(
    df: pd.DataFrame,
    roi_col: str = "ROI_annotation",
    prefix: str = "ROI__",
    group_regex: Optional[str] = None,
) -> pd.DataFrame:
    """Create binary indicator columns from ROI annotation labels.

    Parameters
    ----------
    df          : pixel DataFrame containing roi_col
    roi_col     : column with per-pixel ROI label (or NA)
    prefix      : prefix for generated binary columns
    group_regex : optional regex with ONE capture group to map full ROI labels to
                  group names before binarising.  E.g.
                  ``"(?:Ctrl_|HDM_)?([A-Za-z]+)"`` maps
                  ``"Ctrl_bronch_01"`` → ``"bronch"`` and
                  ``"HDM_Bronch_06"`` → ``"Bronch"``.
                  When None the full label is used as-is (one column per ROI).
    """
    if roi_col not in df.columns:
        return df

    labels = df[roi_col].astype(object)  # keep NA as NaN

    if group_regex is not None:
        def _extract(val):
            if pd.isna(val):
                return val
            m = re.search(group_regex, str(val), flags=re.IGNORECASE)
            return m.group(1).lower() if m else val  # lowercase for case-insensitive grouping

        labels = labels.map(_extract)
        # Report the mapping for transparency
        mapping = (
            df[[roi_col]]
            .assign(_group=labels)
            .dropna(subset=[roi_col])
            .drop_duplicates()
            .sort_values(roi_col)
        )
        print(f"[roi] group_regex mapping ({len(mapping)} unique labels → "
              f"{labels.nunique(dropna=True)} groups):")
        for _, row in mapping.iterrows():
            print(f"       {row[roi_col]!r:40s} → {row['_group']!r}")

    dummies = pd.get_dummies(labels.astype("category"), prefix=prefix, prefix_sep="", dtype=int)
    return pd.concat([df, dummies], axis=1)


def add_roi_distance_columns(
    df: pd.DataFrame,
    roi_bin_cols: Sequence[str],
    x_col: str = "x",
    y_col: str = "y",
    radius: float = 10.0,
    decay: str = "linear",      # "linear" | "gaussian" | "binary"
    suffix: str = "_decay",
    sample_col: str = "sample",
) -> pd.DataFrame:
    """Add continuous distance-decay columns alongside each ROI binary column.

    For each binary ROI column (0/1), creates ``<col><suffix>`` where:

    * Pixels **inside** the ROI → 1.0
    * Pixels **outside within radius** → decayed value in (0, 1)
    * Pixels **beyond radius** → 0.0

    Distance is the Euclidean distance (in the same x/y coordinate units as the
    pixel table — raw grid pixel units for DESI) to the nearest inside-ROI pixel,
    computed per-sample via a KD-tree.

    Decay modes
    -----------
    ``"binary"``
        Step function: 1 inside + within radius, 0 beyond.
    ``"linear"``
        ``1 - d / radius``, clipped to [0, 1].
    ``"gaussian"``
        ``exp(-0.5 * (d / sigma)^2)`` where ``sigma = radius / 3``.
        Hard cutoff at radius so very distant pixels are exactly 0.

    Parameters
    ----------
    radius
        In the same units as x/y (raw grid pixel units for this pipeline).
        DESI pixels are ~1 unit apart in this space, so radius=5 ≈ 5 pixels ≈ 500 µm
        at 100 µm pixel spacing.
    """
    from sklearn.neighbors import KDTree

    df = df.copy()

    for col in roi_bin_cols:
        new_col = col + suffix
        df[new_col] = 0.0

        for sample in df[sample_col].unique():
            mask_sample = df[sample_col] == sample
            g = df.loc[mask_sample]

            inside = g[col].astype(bool)
            xy_inside = g.loc[inside, [x_col, y_col]].to_numpy(float)

            if len(xy_inside) == 0:
                continue

            # Inside pixels always 1.0
            df.loc[g.index[inside], new_col] = 1.0

            # Outside pixels: distance to nearest inside pixel
            outside = ~inside
            if not outside.any():
                continue

            xy_outside = g.loc[outside, [x_col, y_col]].to_numpy(float)
            tree = KDTree(xy_inside)
            dists = tree.query(xy_outside, k=1, return_distance=True)[0].ravel()

            if decay == "binary":
                vals = (dists <= radius).astype(float)
            elif decay == "linear":
                vals = np.clip(1.0 - dists / radius, 0.0, 1.0)
            elif decay == "gaussian":
                sigma = radius / 3.0
                vals = np.exp(-0.5 * (dists / sigma) ** 2)
                vals[dists > radius] = 0.0   # hard cutoff
            else:
                raise ValueError(
                    f"Unknown decay mode: {decay!r}. Choose 'binary', 'linear', or 'gaussian'."
                )

            df.loc[g.index[outside], new_col] = vals

        n_nonzero = (df[new_col] > 0).sum()
        frac = n_nonzero / len(df) * 100
        print(
            f"[roi_halo] {col} → {new_col}: "
            f"{n_nonzero:,}/{len(df):,} pixels > 0 ({frac:.1f}%)  "
            f"radius={radius} decay={decay}"
        )

    return df


def add_cell_halo_columns(
    pixels: pd.DataFrame,
    cell_meta: pd.DataFrame,
    cell_feature_cols: Sequence[str],
    x_col: str = "x",
    y_col: str = "y",
    halo_radius: float = 10.0,
    decay: str = "linear",      # "linear" | "gaussian" | "binary"
    suffix: str = "_halo",
    sample_col: str = "sample",
    cell_x_col: str = "cell_x",
    cell_y_col: str = "cell_y",
    cell_type_col: str = "cell_type",
    effective_radius_col: str = "effective_radius",
) -> pd.DataFrame:
    """Add continuous distance-decay halo columns for each CELL__* binary column.

    For each cell-type binary column (0/1), creates ``<col><suffix>`` where:

    * Pixels **inside** a cell of that type → 1.0
    * Pixels **outside but within halo_radius of any cell edge** → decayed value in (0, 1)
    * Pixels **beyond halo_radius from every cell edge** → 0.0

    Distance is measured from the nearest cell's **edge** (centroid distance minus
    that cell's effective_radius), not from the centroid itself.  This means the
    halo starts at the cell boundary and extends outward by halo_radius units.

    Decay modes — same as add_roi_distance_columns:
    ``"linear"``, ``"gaussian"`` (sigma = halo_radius / 3), ``"binary"``.

    Parameters
    ----------
    halo_radius
        In the same units as x/y (raw grid pixel units for DESI).
    cell_feature_cols
        The CELL__* binary column names already present in pixels (from
        pixel_celltype_features).  A halo column is added for each.
    """
    from sklearn.neighbors import KDTree

    pixels = pixels.copy()

    for col in cell_feature_cols:
        # Recover cell type name by stripping the prefix
        # col looks like "CELL__Macrophage"
        cell_type = col.split("__", 1)[-1] if "__" in col else col
        new_col = col + suffix
        pixels[new_col] = 0.0

        for sample in pixels[sample_col].unique():
            mask_sample = pixels[sample_col] == sample
            g = pixels.loc[mask_sample]

            # Cells of this type in this sample
            if "sample" in cell_meta.columns:
                cm = cell_meta.loc[cell_meta["sample"] == sample]
            else:
                cm = cell_meta
            cm_type = cm.loc[cm[cell_type_col].astype(str) == cell_type]

            if cm_type.empty:
                continue

            cell_xy = cm_type[[cell_x_col, cell_y_col]].to_numpy(float)
            cell_radii = cm_type[effective_radius_col].to_numpy(float)

            # Pixels already inside → 1.0
            inside_mask = g[col].astype(bool)
            pixels.loc[g.index[inside_mask], new_col] = 1.0

            # Pixels outside → measure distance to nearest cell edge
            outside_mask = ~inside_mask
            if not outside_mask.any():
                continue

            pix_xy = g.loc[outside_mask, [x_col, y_col]].to_numpy(float)
            tree = KDTree(cell_xy)

            # Distance to nearest centroid + index of that centroid
            dists_centroid, idx_nearest = tree.query(pix_xy, k=1)
            dists_centroid = dists_centroid.ravel()
            idx_nearest = idx_nearest.ravel()

            # Distance beyond the cell edge  (can be negative if inside — clamp to 0)
            edge_dist = np.maximum(0.0, dists_centroid - cell_radii[idx_nearest])

            if decay == "binary":
                vals = (edge_dist <= halo_radius).astype(float)
            elif decay == "linear":
                vals = np.clip(1.0 - edge_dist / halo_radius, 0.0, 1.0)
            elif decay == "gaussian":
                sigma = halo_radius / 3.0
                vals = np.exp(-0.5 * (edge_dist / sigma) ** 2)
                vals[edge_dist > halo_radius] = 0.0
            else:
                raise ValueError(
                    f"Unknown decay mode: {decay!r}. Choose 'binary', 'linear', or 'gaussian'."
                )

            pixels.loc[g.index[outside_mask], new_col] = vals

        n_nonzero = (pixels[new_col] > 0).sum()
        frac = n_nonzero / len(pixels) * 100
        print(
            f"[cell_halo] {col} → {new_col}: "
            f"{n_nonzero:,}/{len(pixels):,} pixels > 0 ({frac:.1f}%)  "
            f"halo_radius={halo_radius} decay={decay}"
        )

    return pixels


def validate_halo_radius(
    pixels: pd.DataFrame,
    radius: float,
    physical_pixel_size_um: float,
    label: str = "halo",
    min_mm: float = 0.05,
    max_mm: float = 10.0,
    sample_n: int = 5000,
    sample_col: str = "sample",
    x_col: str = "x",
    y_col: str = "y",
) -> float:
    """Validate a halo/decay radius against the pixel coordinate spacing.

    Computes the median nearest-neighbour distance between pixels (in coordinate
    units) for the first sample, converts the supplied radius to millimetres
    using ``physical_pixel_size_um``, and raises ``ValueError`` if the result
    falls outside [min_mm, max_mm].

    Returns the estimated radius in millimetres so the caller can log it.

    Parameters
    ----------
    radius
        Radius in the same coordinate units as pixel x/y.
    physical_pixel_size_um
        Physical size of one coordinate unit in micrometres.
        For this pipeline (raw grid centroids, 1 unit = 1 DESI pixel): use 100.
    label
        Name used in error/print messages (e.g. "roi_halo", "cell_halo").
    min_mm, max_mm
        Acceptable physical radius range in millimetres.
    sample_n
        Max pixels to subsample for the NN distance estimate.
    """
    from sklearn.neighbors import NearestNeighbors

    # Use the first sample only — spacing is the same across samples
    first_sample = pixels[sample_col].iloc[0]
    g = pixels.loc[pixels[sample_col] == first_sample, [x_col, y_col]].to_numpy(float)

    if len(g) > sample_n:
        rng = np.random.default_rng(0)
        g = g[rng.choice(len(g), size=sample_n, replace=False)]

    nn = NearestNeighbors(n_neighbors=2).fit(g)
    dists, _ = nn.kneighbors(g)
    median_spacing = float(np.median(dists[:, 1]))  # nearest neighbour (col 0 is self = 0)

    radius_mm = radius * median_spacing * physical_pixel_size_um / 1000.0

    print(
        f"[{label}] coordinate spacing ≈ {median_spacing:.3f} units  "
        f"({physical_pixel_size_um} µm/unit)  →  "
        f"radius {radius} units ≈ {radius_mm:.2f} mm"
    )

    if not (min_mm <= radius_mm <= max_mm):
        raise ValueError(
            f"[{label}] radius={radius} units resolves to {radius_mm:.3f} mm, "
            f"which is outside the expected range [{min_mm}, {max_mm}] mm.\n"
            f"  coordinate spacing ≈ {median_spacing:.3f} units  "
            f"  physical_pixel_size_um = {physical_pixel_size_um}\n"
            f"  If your coordinates are in µm rather than pixel-index units, "
            f"set physical_pixel_size_um: 1 in your config and increase radius accordingly.\n"
            f"  To override this check, adjust min_mm / max_mm in the config."
        )

    return radius_mm


def validate_cell_coordinate_space(
    pixels: pd.DataFrame,
    cell_meta: pd.DataFrame,
    sample_col: str = "sample",
    x_col: str = "x",
    y_col: str = "y",
    cell_x_col: str = "cell_x",
    cell_y_col: str = "cell_y",
    max_range_ratio: float = 20.0,
) -> None:
    """Raise ValueError if cell and pixel coordinates appear to be in different units.

    Compares the x/y range of pixels vs cells within the same sample.
    If cell range is more than ``max_range_ratio`` × larger than pixel range,
    it is almost certainly in a different coordinate space (e.g. µm vs pixel-index).
    """
    for sample in pixels[sample_col].unique():
        pix = pixels.loc[pixels[sample_col] == sample, [x_col, y_col]]
        if "sample" in cell_meta.columns:
            cel = cell_meta.loc[cell_meta["sample"] == sample, [cell_x_col, cell_y_col]]
        else:
            cel = cell_meta[[cell_x_col, cell_y_col]]

        if cel.empty:
            continue

        pix_xrange = float(pix[x_col].max() - pix[x_col].min())
        cel_xrange = float(cel[cell_x_col].max() - cel[cell_x_col].min())

        if pix_xrange == 0:
            continue

        ratio = cel_xrange / pix_xrange
        print(
            f"[cell_coords] sample={sample}  "
            f"pixel x-range={pix_xrange:.1f}  cell x-range={cel_xrange:.1f}  "
            f"ratio={ratio:.1f}"
        )

        if ratio > max_range_ratio or ratio < 1.0 / max_range_ratio:
            raise ValueError(
                f"[cell_coords] Cell and pixel coordinates appear to be in DIFFERENT units "
                f"for sample='{sample}'.\n"
                f"  pixel x-range = {pix_xrange:.1f} units\n"
                f"  cell  x-range = {cel_xrange:.1f} units  (ratio = {ratio:.1f})\n"
                f"  Halo radius values will be meaningless across these spaces.\n"
                f"  Check that sd.transform() is putting cells into the same "
                f"coordinate system as the pixel grid centroids."
            )


# ----------------------------
# Pair / sample sets
# ----------------------------

def make_upper_triangle_pairs(cols: Sequence[str]) -> List[Tuple[str, str]]:
    cols = list(cols)
    return [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]

def make_pairs(a: Sequence[str], b: Sequence[str], allow_self: bool = False) -> List[Tuple[str, str]]:
    pairs = []
    for x in a:
        for y in b:
            if (not allow_self) and (x == y):
                continue
            pairs.append((x, y))
    return pairs

def make_sample_sets(samples: Sequence[str], mode: str, custom: Optional[List[List[str]]] = None) -> List[List[str]]:
    samples = list(dict.fromkeys(list(samples)))
    if mode == "custom":
        if not custom:
            raise ValueError("mode='custom' requires custom sample sets.")
        return [list(s) for s in custom]
    sets: List[List[str]] = []
    if mode in ("each", "each+all"):
        sets.extend([[s] for s in samples])
    if mode in ("all", "each+all"):
        sets.append(samples)
    if mode == "pairwise":
        sets.extend([list(p) for p in itertools.combinations(samples, 2)])
    return sets


def find_table_by_regex(sdata: SpatialData, name_regex: str) -> Optional[str]:
    keys = list(getattr(sdata, "tables", {}).keys())
    for k in keys:
        if re.search(name_regex, k, re.IGNORECASE):
            return k
    return None

def pick_first_existing_col_df(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate column name that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def pick_first_existing_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate that exists in a column list, or None."""
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def find_shapes_by_regex(sdata: SpatialData, name_regex: str) -> List[str]:
    keys = list(getattr(sdata, "shapes", {}).keys())
    return sorted([k for k in keys if re.search(name_regex, k, re.IGNORECASE)])


def align_series_to_shapes(
    cell_gdf_t: "gpd.GeoDataFrame",
    adata,
    value_col: str,
) -> pd.Series:
    """Align a table obs column to a transformed cell shapes GeoDataFrame.

    Tries:
      1) direct match: shapes index ~ obs_names
      2) ID column match (cell_id/Object_ID/etc.)
      3) NN fallback on x/y columns in table
    """
    shape_idx = pd.Index(cell_gdf_t.index.astype(str))
    obs_names = pd.Index(adata.obs_names.astype(str))

    # Case 1: direct match
    if shape_idx.isin(obs_names).mean() > 0.98:
        out = adata.obs.loc[shape_idx, value_col]
        out.index = shape_idx
        return out

    # Case 2: id columns
    candidate_id_cols = ["cell_id", "Cell_ID", "object_id", "Object_ID", "roi", "ROI", "Number", "id"]
    id_col = next((c for c in candidate_id_cols if c in adata.obs.columns), None)
    if id_col is not None:
        map_df = (
            pd.DataFrame({
                "id": adata.obs[id_col].astype(str).values,
                "val": adata.obs[value_col].values,
            })
            .drop_duplicates("id")
        )
        if shape_idx.isin(map_df["id"]).mean() > 0.9:
            out = pd.Series(shape_idx, index=shape_idx, name="id").to_frame()
            out = out.merge(map_df, on="id", how="left")["val"]
            out.index = shape_idx
            return out

    # Case 3: NN fallback on x/y in obs
    if ("x" in adata.obs.columns) and ("y" in adata.obs.columns):
        tab_xy = adata.obs[["x", "y"]].to_numpy(float)
        shp_xy = np.c_[
            cell_gdf_t.geometry.centroid.x.to_numpy(float),
            cell_gdf_t.geometry.centroid.y.to_numpy(float),
        ]
        tree = KDTree(tab_xy)
        _, nn = tree.query(shp_xy, k=1)
        nn = nn.flatten()
        out = adata.obs.iloc[nn][value_col]
        out.index = shape_idx
        return out

    raise KeyError(f"Could not align '{value_col}' to shapes.")

def build_pixel_to_cell_radius_map_spatialdata(
    sdata: SpatialData,
    pixels_df: pd.DataFrame,
    sample: str,
    cell_shapes_name: str,
    target_coordinate_system: str,
    cell_type_table_name: str,
    cell_type_col: str,
    cell_area_table_name: str,
    cell_area_col: str,
    rm_cell_types: Optional[Sequence[str]] = None,
    cell_radius_scalar: float = 1.0,
    progress_every_pixels: int = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Coordinate-system aware pixel->cell radius assignment (notebook-derived).

    Uses:
      - sd.transform(cell shapes -> target_coordinate_system)
      - cell type/area pulled from sdata tables, aligned to shapes
      - KDTree radius containment assignment (many-to-many)

    Returns:
      pixel_cell: columns [pixel_id, cell_idx, sample]
      cell_meta:  columns [cell_idx, cell_type, cell_area, cell_radius, effective_radius, cell_x, cell_y, sample]
    """
    # transform cell shapes into the CS used by pixels for this sample
    if gpd is None:
        raise ImportError("geopandas required for cell shape operations.")
    cell_gdf_t = sd.transform(sdata.shapes[cell_shapes_name], to_coordinate_system=target_coordinate_system)

    # pixel xy for this sample (already in sample CS from grid centroids)
    g = pixels_df.loc[pixels_df["sample"] == sample]
    desi_xy = g[["x", "y"]].to_numpy(float)

    # cell centroids
    cell_xy_all = np.c_[
        cell_gdf_t.geometry.centroid.x.to_numpy(float),
        cell_gdf_t.geometry.centroid.y.to_numpy(float),
    ]

    # pull tables
    ad_ct = sdata.tables[cell_type_table_name]
    ad_ar = sdata.tables[cell_area_table_name]

    cell_types_all = align_series_to_shapes(cell_gdf_t, ad_ct, cell_type_col).astype(str).fillna("NA")
    cell_area_all = pd.to_numeric(align_series_to_shapes(cell_gdf_t, ad_ar, cell_area_col), errors="coerce").to_numpy(float)

    keep_mask = np.ones(len(cell_gdf_t), dtype=bool)
    if rm_cell_types:
        rm_set = set(map(str, rm_cell_types))
        keep_mask &= ~cell_types_all.astype(str).isin(rm_set).to_numpy()

    cell_xy = cell_xy_all[keep_mask]
    cell_types = cell_types_all.to_numpy()[keep_mask].astype(str)
    cell_area = cell_area_all[keep_mask].astype(float)

    cell_radius = np.sqrt(cell_area / np.pi)
    effective_radius = float(cell_radius_scalar) * cell_radius

    cell_meta = pd.DataFrame({
        "cell_idx": np.arange(cell_xy.shape[0], dtype=int),
        "cell_type": cell_types,
        "cell_area": cell_area,
        "cell_radius": cell_radius,
        "effective_radius": effective_radius,
        "cell_x": cell_xy[:, 0],
        "cell_y": cell_xy[:, 1],
        "sample": sample,
    })

    # Build containment mapping
    tree = KDTree(cell_xy)
    max_r = np.nanmax(effective_radius[np.isfinite(effective_radius)]) if cell_xy.shape[0] else 0.0
    cand_lists = tree.query_radius(desi_xy, r=max_r)

    rows = []
    for pix_i, cand in enumerate(cand_lists):
        if progress_every_pixels and (pix_i % int(progress_every_pixels) == 0) and pix_i > 0:
            print(f"[cell] sample={sample} processed {pix_i:,}/{desi_xy.shape[0]:,} pixels...")
        if len(cand) == 0:
            continue
        d = np.linalg.norm(cell_xy[cand] - desi_xy[pix_i], axis=1)
        inside = cand[d <= effective_radius[cand]]
        for c in inside:
            rows.append((g.index[pix_i], int(c), sample))

    pixel_cell = pd.DataFrame(rows, columns=["pixel_id", "cell_idx", "sample"])
    return pixel_cell, cell_meta


def build_pixel_to_cell_radius_map_from_table(
    pixels_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    cell_x_col: str,
    cell_y_col: str,
    cell_area_col: str,
    cell_type_col: str,
    rm_cell_types: Optional[Sequence[str]] = None,
    cell_radius_scalar: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    cell_df must have centroid x/y + area + type.
    Computes radius from area, scales it, then assigns pixels to cells by distance.
    Returns:
      pixel_cell: [pixel_id, cell_idx]
      cell_meta:  [cell_idx, cell_type, cell_area, cell_radius, effective_radius, cell_x, cell_y]
    """
    cdf = cell_df.copy()

    # remove unwanted types
    if rm_cell_types:
        keep_mask = ~cdf[cell_type_col].astype(str).isin(list(rm_cell_types))
        print(f"Removing {(~keep_mask).sum():,} unwanted cells")
        print(f"Keeping {keep_mask.sum():,}/{len(keep_mask):,} cells")
        cdf = cdf.loc[keep_mask].copy()

    cell_xy = cdf[[cell_x_col, cell_y_col]].to_numpy(float)
    cell_area = cdf[cell_area_col].to_numpy(float)
    cell_types = cdf[cell_type_col].astype(str).to_numpy()

    cell_radius = np.sqrt(cell_area / np.pi)
    effective_radius = float(cell_radius_scalar) * cell_radius

    cell_meta = pd.DataFrame({
        "cell_idx": np.arange(len(cdf), dtype=int),
        "cell_type": cell_types,
        "cell_area": cell_area,
        "cell_radius": cell_radius,
        "effective_radius": effective_radius,
        "cell_x": cell_xy[:, 0],
        "cell_y": cell_xy[:, 1],
    })

    desi_xy = pixels_df[["x", "y"]].to_numpy(float)

    tree = KDTree(cell_xy)
    finite_radii = effective_radius[np.isfinite(effective_radius)]
    if len(finite_radii) == 0:
        return pd.DataFrame(columns=["pixel_index", "cell_index", "distance"])
    max_r = np.nanmax(finite_radii)
    cand_lists = tree.query_radius(desi_xy, r=max_r)

    rows = []
    eff = effective_radius
    for pix_i, cand in enumerate(cand_lists):
        if len(cand) == 0:
            continue
        d = np.linalg.norm(cell_xy[cand] - desi_xy[pix_i], axis=1)
        inside = cand[d <= eff[cand]]
        for c in inside:
            rows.append((pixels_df.index[pix_i], int(c)))

    pixel_cell = pd.DataFrame(rows, columns=["pixel_id", "cell_idx"])
    return pixel_cell, cell_meta

def pixel_celltype_features(
    pixels_df: pd.DataFrame,
    pixel_cell: pd.DataFrame,
    cell_meta: pd.DataFrame,
    mode: str = "onehot_any",   # onehot_any | counts
    prefix: str = "CELL__",
) -> Tuple[pd.DataFrame, List[str]]:
    if pixel_cell.empty:
        return pixels_df, []

    merged = pixel_cell.merge(cell_meta[["cell_idx", "cell_type"]], on="cell_idx", how="left")
    ct = merged.groupby(["pixel_id", "cell_type"]).size().unstack(fill_value=0)

    if mode == "onehot_any":
        ct = (ct > 0).astype(int)

    ct.columns = [f"{prefix}{c}" for c in ct.columns]

    out = pixels_df.join(ct, how="left")
    out[ct.columns] = out[ct.columns].fillna(0).astype(int if mode == "onehot_any" else float)
    return out, list(ct.columns)
    
# ----------------------------
# Spearman
# ----------------------------

def spearman_pairs(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    sample_sets: Sequence[Sequence[str]],
    sample_col: str = "sample",
) -> pd.DataFrame:
    rows = []
    for ss in sample_sets:
        ss = list(ss)
        label = "+".join(ss)
        g = df[df[sample_col].isin(ss)]
        for a, b in pairs:
            x = g[a].to_numpy(float)
            y = g[b].to_numpy(float)
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                continue
            rho, p = spearmanr(x, y, nan_policy="omit")
            rows.append(dict(method="spearman", sample_set=label, x_var=a, y_var=b,
                             n=int(np.sum(~np.isnan(x) & ~np.isnan(y))), stat=float(rho), p=float(p)))
    return pd.DataFrame(rows)


def pearson_pairs(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    sample_sets: Sequence[Sequence[str]],
    sample_col: str = "sample",
) -> pd.DataFrame:
    """Compute Pearson r for each pair across each sample set."""
    rows = []
    for ss in sample_sets:
        ss = list(ss)
        label = "+".join(ss)
        g = df[df[sample_col].isin(ss)]
        for a, b in pairs:
            x = g[a].to_numpy(float)
            y = g[b].to_numpy(float)
            # drop rows where either value is NaN
            mask = ~np.isnan(x) & ~np.isnan(y)
            x, y = x[mask], y[mask]
            if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
                continue
            r, p = pearsonr(x, y)
            rows.append(dict(method="pearson", sample_set=label, x_var=a, y_var=b,
                             n=int(len(x)), stat=float(r), p=float(p)))
    return pd.DataFrame(rows)


# ----------------------------
# Moran weights
# ----------------------------

_log = logging.getLogger(__name__)

# Module-level in-memory cache: {cache_key: (w, meta)}
# Survives for the lifetime of the Python process so repeated calls within one
# pipeline run (e.g. univariate then bivariate on the same sample) reuse the
# already-built weights object without any disk I/O.
_WEIGHTS_CACHE: Dict[str, Any] = {}


def _weights_cache_key(xy: np.ndarray, wtype: str, threshold: Optional[float],
                        threshold_multiplier: float, knn_k: int, binary: bool) -> str:
    """Stable hash key for a weights configuration."""
    h = hashlib.md5()
    h.update(xy.tobytes())
    h.update(f"{wtype}|{threshold}|{threshold_multiplier}|{knn_k}|{binary}".encode())
    return h.hexdigest()


def _estimate_distance_threshold(xy: np.ndarray, sample_size: int = 5000) -> float:
    """Estimate a suitable distance threshold as median nearest-neighbour distance.

    For large arrays a random subsample of `sample_size` points is used to
    keep runtime manageable.
    """
    from sklearn.neighbors import NearestNeighbors
    n = xy.shape[0]
    if n > sample_size:
        idx = np.random.default_rng(0).choice(n, size=sample_size, replace=False)
        xy = xy[idx]
    nn = NearestNeighbors(n_neighbors=2).fit(xy)
    dists, _ = nn.kneighbors(xy)
    return float(np.median(dists[:, 1]) * 1.01)


def build_weights(
    xy: np.ndarray,
    wtype: str = "distanceband",
    threshold: Optional[float] = None,
    threshold_multiplier: float = 1.0,
    knn_k: int = 8,
    binary: bool = True,
    cache_dir: Optional[str] = None,   # if set, persist weights to disk as well
) -> Tuple[Any, Dict]:
    """Build (or retrieve from cache) a libpysal spatial weights object.

    Caching strategy
    ----------------
    1. Check the in-process memory cache first — zero cost if already built this
       session (e.g. univariate + bivariate called on the same sample/config).
    2. If ``cache_dir`` is provided, look for a pickled file on disk — useful
       when re-running only part of the pipeline or across Python sessions.
    3. Build from scratch, store in both caches.

    The cache key is an MD5 hash of the coordinate array bytes + config string,
    so different samples or multiplier values never collide.
    """
    key = _weights_cache_key(xy, wtype, threshold, threshold_multiplier, knn_k, binary)

    # ── 1. in-process memory cache ────────────────────────────────────────────
    if key in _WEIGHTS_CACHE:
        _log.debug("build_weights: memory cache hit (%s)", key[:8])
        return _WEIGHTS_CACHE[key]

    # ── 2. disk cache ─────────────────────────────────────────────────────────
    if cache_dir is not None:
        import os
        os.makedirs(cache_dir, exist_ok=True)
        disk_path = os.path.join(cache_dir, f"weights_{key[:16]}.pkl")
        if os.path.exists(disk_path):
            _log.info("build_weights: disk cache hit  → %s", disk_path)
            with open(disk_path, "rb") as fh:
                result = pickle.load(fh)
            _WEIGHTS_CACHE[key] = result
            return result

    # ── 3. build ──────────────────────────────────────────────────────────────
    _log.info("build_weights: building %s (n=%d) …", wtype, len(xy))
    t0 = time.time()

    if wtype.lower() == "distanceband":
        if threshold is None:
            threshold = _estimate_distance_threshold(xy)
        threshold = float(threshold) * float(threshold_multiplier)
        w = DistanceBand(xy, threshold=threshold, binary=binary, silence_warnings=True)
        w.transform = "R"
        meta = {"type": "distanceband", "threshold": threshold, "binary": binary}

    elif wtype.lower() == "knn":
        w = KNN.from_array(xy, k=int(knn_k))
        w.transform = "R"
        meta = {"type": "knn", "k": int(knn_k)}

    else:
        raise ValueError(f"Unknown weights.type={wtype!r}")

    elapsed = time.time() - t0
    _log.info("build_weights: done in %.1fs  (mean neighbours/pixel: %.1f)",
              elapsed, np.mean([len(v) for v in w.neighbors.values()]))

    result = (w, meta)
    _WEIGHTS_CACHE[key] = result

    if cache_dir is not None:
        with open(disk_path, "wb") as fh:
            pickle.dump(result, fh, protocol=4)
        _log.info("build_weights: saved to disk → %s", disk_path)

    return result


def clear_weights_cache() -> None:
    """Evict all in-process cached weights (call if memory is a concern)."""
    _WEIGHTS_CACHE.clear()
    _log.info("build_weights: in-process cache cleared")

def _moran_extract(mi_obj) -> Dict[str, float]:
    I = float(getattr(mi_obj, "I"))
    p_norm = float(getattr(mi_obj, "p_norm", np.nan))
    z_norm = getattr(mi_obj, "z_norm", None)
    z_norm = float(z_norm) if z_norm is not None else (float(norm.isf(p_norm)) * np.sign(I) if np.isfinite(p_norm) else np.nan)

    # permutation stats if present
    p_sim = getattr(mi_obj, "p_sim", None)
    z_sim = getattr(mi_obj, "z_sim", None)
    p_sim = float(p_sim) if p_sim is not None else np.nan
    z_sim = float(z_sim) if z_sim is not None else np.nan

    # variance/se if available
    var = None
    for attr in ("VI_norm", "VI_rand", "VI_sim", "VI"):
        if hasattr(mi_obj, attr):
            try:
                var = float(getattr(mi_obj, attr))
                break
            except Exception:
                pass
    se = np.nan
    if var is not None and var > 0:
        se = float(np.sqrt(var))
    elif np.isfinite(z_norm) and z_norm != 0:
        se = float(abs(I / z_norm))

    return dict(I=I, p_norm=p_norm, z_norm=z_norm, p_sim=p_sim, z_sim=z_sim, se=se)

def moran_univariate_per_sample(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    sample_col: str = "sample",
    x_col: str = "x",
    y_col: str = "y",
    wtype: str = "distanceband",
    threshold: Optional[float] = None,
    threshold_multiplier: float = 1.0,
    knn_k: int = 8,
    binary: bool = True,
    permutations: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    if Moran is None:
        raise ImportError("esda required for Moran.")
    rows = []
    n_cols = len(value_cols)
    for sample, g in df.groupby(sample_col, sort=True):
        xy = g[[x_col, y_col]].to_numpy(float)
        w, _ = build_weights(
            xy, wtype=wtype, threshold=threshold,
            threshold_multiplier=threshold_multiplier, knn_k=knn_k,
            binary=binary, cache_dir=cache_dir)
        for i, c in enumerate(value_cols, 1):
            print(f"  [univariate moran] {sample}  {i}/{n_cols}: {c}", flush=True)
            v = g[c].to_numpy(float)
            if np.all(np.isnan(v)) or np.nanstd(v) == 0:
                continue
            v = np.nan_to_num(v, nan=np.nanmean(v))
            mi = Moran(v, w, two_tailed=False, permutations=permutations) if permutations else Moran(v, w, two_tailed=False)
            ext = _moran_extract(mi)
            rows.append(dict(method="moran_univariate", sample=sample, x_var=c, y_var=None, n=len(v), **ext))
    return pd.DataFrame(rows)

def moran_bivariate_per_sample(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    sample_col: str = "sample",
    x_col: str = "x",
    y_col: str = "y",
    wtype: str = "distanceband",
    threshold: Optional[float] = None,
    threshold_multiplier: float = 1.0,
    knn_k: int = 8,
    binary: bool = True,
    permutations: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    if Moran_BV is None:
        raise ImportError("esda required for Moran_BV.")
    rows = []
    n_pairs = len(pairs)
    for sample, g in df.groupby(sample_col, sort=True):
        xy = g[[x_col, y_col]].to_numpy(float)
        w, _ = build_weights(
            xy, wtype=wtype, threshold=threshold,
            threshold_multiplier=threshold_multiplier, knn_k=knn_k,
            binary=binary, cache_dir=cache_dir)
        for i, (a, b) in enumerate(pairs, 1):
            print(f"  [bivariate moran] {sample}  {i}/{n_pairs}: {a} ~ {b}", flush=True)
            x = g[a].to_numpy(float)
            y = g[b].to_numpy(float)
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                continue
            x = np.nan_to_num(x, nan=np.nanmean(x))
            y = np.nan_to_num(y, nan=np.nanmean(y))
            mbv = Moran_BV(x, y, w, permutations=permutations) if permutations else Moran_BV(x, y, w)
            ext = _moran_extract(mbv)
            rows.append(dict(method="moran_bivariate", sample=sample, x_var=a, y_var=b, n=len(x), **ext))
    return pd.DataFrame(rows)


# ----------------------------
# Combining per-sample Moran (weighted mean I + Fisher + Stouffer + RE)
# ----------------------------

def combine_moran_effects(
    per_sample: pd.DataFrame,
    group_cols: Sequence[str],
    p_source: str = "p_norm",  # or "p_sim"
    z_source: str = "z_norm",  # or "z_sim" (often present if permutations)
    stouffer_weight: str = "sqrt_n",  # sqrt_n | n | equal | inv_se
) -> pd.DataFrame:
    # Auto-fallback if requested columns are missing or all-NaN
    if (p_source not in per_sample.columns or per_sample[p_source].isna().all()) \
            and "p_sim" in per_sample.columns:
        p_source = "p_sim"
    if (z_source not in per_sample.columns or per_sample[z_source].isna().all()) \
            and "z_sim" in per_sample.columns:
        z_source = "z_sim"

    if p_source not in per_sample.columns or z_source not in per_sample.columns:
        raise ValueError(f"Requested p/z sources not found: {p_source}, {z_source}")

    rows = []
    for key, g in per_sample.groupby(list(group_cols), sort=False):
        g = g.dropna(subset=["I", p_source, z_source], how="any").copy()
        if g.empty:
            continue

        n = g["n"].to_numpy(float)
        I = g["I"].to_numpy(float)

        # size-weighted mean I
        w_size = n / np.sum(n)
        I_size_weighted = float(np.sum(w_size * I))

        # Fisher
        p = np.clip(g[p_source].to_numpy(float), 1e-300, 1.0)
        chi_stat = float(-2.0 * np.sum(np.log(p)))
        df_chi = 2 * len(p)
        p_fisher = float(1.0 - chi2.cdf(chi_stat, df=df_chi))

        # Stouffer
        z = g[z_source].to_numpy(float)
        if stouffer_weight == "sqrt_n":
            w = np.sqrt(n)
        elif stouffer_weight == "n":
            w = n
        elif stouffer_weight == "inv_se":
            se = g.get("se", pd.Series([np.nan] * len(g))).to_numpy(float)
            w = np.where(np.isfinite(se) & (se > 0), 1.0 / se, 0.0)
            if np.all(w == 0):
                w = np.sqrt(n)
        else:
            w = np.ones_like(z)

        z_st = float(np.sum(w * z) / np.sqrt(np.sum(w * w))) if np.sum(w * w) > 0 else float(np.mean(z))
        p_stouffer = float(2.0 * (1.0 - norm.cdf(abs(z_st))))

        # Random-effects (DerSimonian–Laird) on I using se if available
        I_re = se_re = z_re = p_re = tau2 = np.nan
        if "se" in g.columns and g["se"].notna().any():
            se = g["se"].to_numpy(float)
            mask = np.isfinite(se) & (se > 0)
            if np.sum(mask) >= 2:
                I_m = I[mask]
                se_m = se[mask]
                v = se_m**2
                w_fe = 1.0 / v
                I_fe = np.sum(w_fe * I_m) / np.sum(w_fe)
                Q = np.sum(w_fe * (I_m - I_fe) ** 2)
                dfQ = len(I_m) - 1
                c = np.sum(w_fe) - (np.sum(w_fe ** 2) / np.sum(w_fe))
                tau2 = float(max(0.0, (Q - dfQ) / c)) if c > 0 else 0.0
                w_re = 1.0 / (v + tau2)
                I_re = float(np.sum(w_re * I_m) / np.sum(w_re))
                se_re = float(np.sqrt(1.0 / np.sum(w_re)))
                z_re = float(I_re / se_re) if se_re > 0 else np.nan
                p_re = float(2.0 * (1.0 - norm.cdf(abs(z_re)))) if np.isfinite(z_re) else np.nan

        out = dict(
            k_samples=int(g.shape[0]),
            p_source=p_source,
            z_source=z_source,
            I_size_weighted=I_size_weighted,
            p_fisher=p_fisher,
            z_stouffer=z_st,
            p_stouffer=p_stouffer,
            I_random_effects=I_re,
            se_random_effects=se_re,
            z_random_effects=z_re,
            p_random_effects=p_re,
            tau2=tau2,
        )

        if isinstance(key, tuple):
            for col, val in zip(group_cols, key):
                out[col] = val
        else:
            out[group_cols[0]] = key

        rows.append(out)

    return pd.DataFrame(rows)

# ----------------------------
# Local moran Univariate
# -----------------------
def local_moran_univariate_summary_per_sample(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    sample_col: str = "sample",
    x_col: str = "x",
    y_col: str = "y",
    wtype: str = "distanceband",
    threshold: Optional[float] = None,
    threshold_multiplier: float = 1.0,
    knn_k: int = 8,
    binary: bool = True,
    permutations: Optional[int] = None,
    alpha: float = 0.05,
    fdr_on_local_p: bool = True,
    p_source: str = "auto",  # auto|p_sim|p_z_sim|p_norm
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    if Moran_Local is None:
        raise ImportError("esda Moran_Local not available; please update esda.")

    rows = []
    n_cols = len(value_cols)
    for sample, g in df.groupby(sample_col, sort=True):
        xy = g[[x_col, y_col]].to_numpy(float)
        w, wmeta = build_weights(
            xy, wtype=wtype, threshold=threshold, threshold_multiplier=threshold_multiplier,
            knn_k=knn_k, binary=binary, cache_dir=cache_dir
        )

        for i, c in enumerate(value_cols, 1):
            print(f"  [local moran uni] {sample}  {i}/{n_cols}: {c}", flush=True)
            x = g[c].to_numpy(float)
            if np.all(np.isnan(x)) or np.nanstd(x) == 0:
                continue
            x = np.nan_to_num(x, nan=np.nanmean(x))

            ml = Moran_Local(x, w, permutations=permutations) if permutations else Moran_Local(x, w)

            # choose p-values
            if p_source == "auto":
                pvals = getattr(ml, "p_sim", None)
                if pvals is None or (not permutations):
                    pvals = getattr(ml, "p_z_sim", None)
                if pvals is None:
                    pvals = getattr(ml, "p_norm", None)
            elif p_source == "p_sim":
                pvals = getattr(ml, "p_sim", None)
            elif p_source == "p_z_sim":
                pvals = getattr(ml, "p_z_sim", None)
            else:
                pvals = getattr(ml, "p_norm", None)

            if pvals is None:
                # last resort: use normal approx from z
                z = getattr(ml, "z_norm", None)
                if z is None:
                    pvals = np.full_like(ml.Is, np.nan, dtype=float)
                else:
                    pvals = 2.0 * (1.0 - norm.cdf(np.abs(z)))

            pvals = np.asarray(pvals, dtype=float)

            # optional FDR across pixels for THIS metabolite within this sample
            if fdr_on_local_p:
                qvals = bh_fdr(pvals)
                sig = qvals <= alpha
            else:
                sig = pvals <= alpha

            q = getattr(ml, "q", None)  # quadrant labels: 1 HH, 2 LH, 3 LL, 4 HL
            if q is None:
                q = np.full_like(sig, 0, dtype=int)
            q = np.asarray(q, dtype=int)

            n = len(sig)
            n_sig = int(np.sum(sig))

            # hotspot/cluster breakdown (only meaningful if q exists)
            hh = int(np.sum(sig & (q == 1)))
            lh = int(np.sum(sig & (q == 2)))
            ll = int(np.sum(sig & (q == 3)))
            hl = int(np.sum(sig & (q == 4)))

            rows.append(dict(
                method="local_moran_univariate_summary",
                sample=sample,
                x_var=c,
                n=n,
                n_sig=n_sig,
                frac_sig=n_sig / n if n else np.nan,
                n_sig_HH=hh, n_sig_LH=lh, n_sig_LL=ll, n_sig_HL=hl,
                frac_sig_HH=hh / n if n else np.nan,
                mean_I_local=float(np.nanmean(ml.Is)),
                mean_I_local_sig=float(np.nanmean(np.where(sig, ml.Is, np.nan))) if n_sig else np.nan,
                weights_type=wmeta.get("type"),
                weights_param=wmeta.get("threshold", wmeta.get("k")),
                permutations=permutations or 0,
                alpha=alpha,
                fdr_on_local_p=fdr_on_local_p,
            ))

    return pd.DataFrame(rows)

# ----------------------------
# Local moran BV
# ----------------------------

def local_moran_bivariate_summary_per_sample(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    sample_col: str = "sample",
    x_col: str = "x",
    y_col: str = "y",
    wtype: str = "distanceband",
    threshold: Optional[float] = None,
    threshold_multiplier: float = 1.0,
    knn_k: int = 8,
    binary: bool = True,
    permutations: Optional[int] = None,
    alpha: float = 0.05,
    fdr_on_local_p: bool = True,
    pixel_out_dir: Optional[str] = None,   # if set, write per-pixel CSVs here
    pixel_index_col: Optional[str] = None, # column to use as pixel identifier (e.g. obs_name)
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Compute local bivariate Moran summary per sample.

    If ``pixel_out_dir`` is provided, also writes one CSV per (sample, pair)
    containing per-pixel local_I, quadrant, p-value — matching the reference
    format with columns:
      pixel_idx, x, y, metA, metB, local_I, p_val, quadrant, quadrant_label, sig
    """
    if Moran_Local_BV is None:
        raise ImportError("esda Moran_Local_BV not available in your esda version.")

    _quadrant_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL", 0: "NS"}

    import os

    summary_rows = []
    for sample, g in df.groupby(sample_col, sort=True):
        xy = g[[x_col, y_col]].to_numpy(float)
        w, wmeta = build_weights(
            xy, wtype=wtype, threshold=threshold, threshold_multiplier=threshold_multiplier,
            knn_k=knn_k, binary=binary, cache_dir=cache_dir
        )

        n_pairs = len(pairs)
        for i, (a, b) in enumerate(pairs, 1):
            print(f"  [local bv moran]  {sample}  {i}/{n_pairs}: {a} ~ {b}", flush=True)
            xv = g[a].to_numpy(float)
            yv = g[b].to_numpy(float)
            if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
                continue
            xv = np.nan_to_num(xv, nan=np.nanmean(xv))
            yv = np.nan_to_num(yv, nan=np.nanmean(yv))

            ml = Moran_Local_BV(xv, yv, w, permutations=permutations) if permutations else Moran_Local_BV(xv, yv, w)

            pvals = getattr(ml, "p_sim", None) if permutations else None
            if pvals is None:
                pvals = getattr(ml, "p_z_sim", None)
            if pvals is None:
                pvals = getattr(ml, "p_norm", None)
            pvals = np.asarray(pvals, dtype=float) if pvals is not None else np.full_like(ml.Is, np.nan, dtype=float)

            if fdr_on_local_p:
                qvals = bh_fdr(pvals)
                sig = qvals <= alpha
            else:
                sig = pvals <= alpha

            q = np.asarray(getattr(ml, "q", np.zeros(len(ml.Is), dtype=int)), dtype=int)
            n = len(sig)
            n_sig = int(np.sum(sig))

            # --- per-pixel CSV ---
            if pixel_out_dir is not None:
                os.makedirs(pixel_out_dir, exist_ok=True)
                # sanitise var names for filename
                def _safe(s):
                    return re.sub(r"[^\w\-]", "_", str(s))
                fname = f"local_bv__{sample}__{_safe(a)}__{_safe(b)}.csv"
                fpath = os.path.join(pixel_out_dir, fname)

                pix_ids = (
                    g[pixel_index_col].to_numpy() if pixel_index_col and pixel_index_col in g.columns
                    else g.index.to_numpy()
                )
                pixel_df = pd.DataFrame({
                    "pixel_idx":     pix_ids,
                    "sample":        sample,
                    "x":             g[x_col].to_numpy(),
                    "y":             g[y_col].to_numpy(),
                    "metA":          a,
                    "metB":          b,
                    "local_I":       ml.Is,
                    "p_val":         pvals,
                    "sig":           sig,
                    "quadrant":      q,
                    "quadrant_label": [_quadrant_labels.get(qi, "NS") for qi in q],
                    "is_HH":         q == 1,
                })
                pixel_df.to_csv(fpath, index=False)
                print(f"[local_bv] wrote {fname}  (n_sig={n_sig}/{n})")

            summary_rows.append(dict(
                method="local_moran_bivariate_summary",
                sample=sample,
                x_var=a,
                y_var=b,
                n=n,
                n_sig=n_sig,
                frac_sig=n_sig / n if n else np.nan,
                mean_I_local=float(np.nanmean(ml.Is)),
                mean_I_local_sig=float(np.nanmean(np.where(sig, ml.Is, np.nan))) if n_sig else np.nan,
                weights_type=wmeta.get("type"),
                weights_param=wmeta.get("threshold", wmeta.get("k")),
                permutations=permutations or 0,
                alpha=alpha,
                fdr_on_local_p=fdr_on_local_p,
            ))

    return pd.DataFrame(summary_rows)

# ----------------------------
# FDR BH
# ----------------------------

def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not np.any(mask):
        return q
    pv = p[mask]
    if _HAS_STATSMODELS:
        qv = multipletests(pv, alpha=0.05, method="fdr_bh")[1]
    else:
        n = pv.size
        order = np.argsort(pv)
        ranked = pv[order]
        qtmp = ranked * n / (np.arange(1, n + 1))
        qtmp = np.minimum.accumulate(qtmp[::-1])[::-1]
        qv = np.empty_like(qtmp)
        qv[order] = np.clip(qtmp, 0, 1)
    q[mask] = qv
    return q

def add_fdr_columns(df: pd.DataFrame, p_cols: Sequence[str], prefix: str = "q_") -> pd.DataFrame:
    out = df.copy()
    for c in p_cols:
        if c in out.columns:
            out[prefix + c] = bh_fdr(out[c].to_numpy(float))
    return out

def _plot_points(ax, x, y, max_points=30000, title=None):
    if len(x) > max_points:
        idx = np.random.default_rng(0).choice(len(x), size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    ax.scatter(x, y, s=1, alpha=0.35)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

def qc_plot_roi_overlay(sdata, pixels, sample, roi_shapes, out_png, max_points=30000):
    roi_name = next((k for k in roi_shapes if k.startswith(sample)), roi_shapes[0])

    # Transform ROI polygons into the sample's coordinate system (physical space)
    roi_gdf_t = sd.transform(sdata.shapes[roi_name], to_coordinate_system=sample)

    # Transform DESI grid into the same coordinate system for pixel positions
    grid_name = next(
        (k for k in sdata.shapes.keys() if k.startswith(sample) and "grid" in k.lower()),
        None,
    )
    g = pixels.loc[pixels["sample"] == sample]
    if grid_name is not None:
        grid_t = sd.transform(sdata.shapes[grid_name], to_coordinate_system=sample)
        px = grid_t.geometry.centroid.x.to_numpy()
        py = grid_t.geometry.centroid.y.to_numpy()
    else:
        # Fallback: use whatever x/y are already in pixels (may be raw coords)
        px = g["x"].to_numpy()
        py = g["y"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 9))
    _plot_points(ax, px, py, max_points=max_points,
                 title=f"{sample} – DESI pixels + ROIs ({roi_name})")

    for _, row in roi_gdf_t.iterrows():
        geom = row.geometry
        name = row.get("name", "")
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            x, y = part.exterior.xy
            ax.plot(x, y, linewidth=1)
        cx, cy = geom.centroid.x, geom.centroid.y
        ax.text(cx, cy, str(name), fontsize=6, ha="center", va="center")

    ax.set_aspect("equal")
    ax.invert_yaxis()  # image coords: y increases downward
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def qc_plot_cell_overlay(sdata, pixels, sample, cell_shape_name, out_png, max_points=30000, n_cells=2000,
                         roi_shapes=None, pixels_before_crop=None):
    # Transform DESI grid to physical CS
    grid_name = next(
        (k for k in sdata.shapes.keys() if k.startswith(sample) and "grid" in k.lower()),
        None,
    )
    g = pixels.loc[pixels["sample"] == sample]
    if grid_name is not None:
        grid_t = sd.transform(sdata.shapes[grid_name], to_coordinate_system=sample)
        # All grid centroids (full set — for "before" panel)
        px_all = grid_t.geometry.centroid.x.to_numpy()
        py_all = grid_t.geometry.centroid.y.to_numpy()
        # Cropped set: only pixels whose obs_name is in the (already-cropped) pixels table
        # (grid index = obs_name; pixel_id is composite sample__obs_name)
        keep_obs = set(g["obs_name"].astype(str).tolist()) if "obs_name" in g.columns else None
        if keep_obs is not None and pixels_before_crop is not None:
            obs_idx = grid_t.index.astype(str)
            mask = np.isin(obs_idx, list(keep_obs))
            px = px_all[mask]
            py = py_all[mask]
        else:
            px = px_all
            py = py_all
    else:
        if pixels_before_crop is not None:
            g_before = pixels_before_crop.loc[pixels_before_crop["sample"] == sample]
            px_all = g_before["x"].to_numpy()
            py_all = g_before["y"].to_numpy()
        else:
            px_all = g["x"].to_numpy()
            py_all = g["y"].to_numpy()
        px = g["x"].to_numpy()
        py = g["y"].to_numpy()

    # Transform cell shapes to physical CS
    cell_gdf_t = sd.transform(sdata.shapes[cell_shape_name], to_coordinate_system=sample)
    cell_cx = cell_gdf_t.geometry.centroid.x.to_numpy()
    cell_cy = cell_gdf_t.geometry.centroid.y.to_numpy()

    # Subsample cells so the plot isn't overwhelmed
    if len(cell_cx) > n_cells:
        idx = np.random.default_rng(0).choice(len(cell_cx), size=n_cells, replace=False)
        cell_cx_sub = cell_cx[idx]
        cell_cy_sub = cell_cy[idx]
    else:
        cell_cx_sub = cell_cx
        cell_cy_sub = cell_cy

    def _add_roi_outlines(ax):
        if roi_shapes:
            roi_name = next((k for k in roi_shapes if k.startswith(sample)), roi_shapes[0])
            roi_gdf_t = sd.transform(sdata.shapes[roi_name], to_coordinate_system=sample)
            for _, row in roi_gdf_t.iterrows():
                geom = row.geometry
                name = row.get("name", "")
                parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
                for part in parts:
                    x, y = part.exterior.xy
                    ax.plot(x, y, linewidth=1, color="red", alpha=0.8)
                cx, cy = geom.centroid.x, geom.centroid.y
                ax.text(cx, cy, str(name), fontsize=6, ha="center", va="center", color="red")

    if pixels_before_crop is not None:
        # Two-panel: before (all pixels) | after (cropped pixels)
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))

        ax0 = axes[0]
        _plot_points(ax0, px_all, py_all, max_points=max_points,
                     title=f"Before crop ({len(px_all):,} px)")
        ax0.scatter(cell_cx_sub, cell_cy_sub, s=2, alpha=0.4, color="orange", label="cells")
        _add_roi_outlines(ax0)
        ax0.set_aspect("equal")
        ax0.invert_yaxis()

        ax1 = axes[1]
        _plot_points(ax1, px, py, max_points=max_points,
                     title=f"After crop ({len(px):,} px)")
        ax1.scatter(cell_cx_sub, cell_cy_sub, s=2, alpha=0.4, color="orange", label="cells")
        _add_roi_outlines(ax1)
        ax1.set_aspect("equal")
        ax1.invert_yaxis()

        fig.suptitle(f"{sample} – DESI pixels + cells ({cell_shape_name})", fontsize=11)
    else:
        # Single panel (no crop)
        fig, ax = plt.subplots(figsize=(9, 9))
        _plot_points(ax, px_all, py_all, max_points=max_points,
                     title=f"{sample} – DESI pixels + cells ({cell_shape_name})")
        ax.scatter(cell_cx_sub, cell_cy_sub, s=2, alpha=0.4, color="orange", label="cells")
        _add_roi_outlines(ax)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
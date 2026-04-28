# corrMSI

Tools for analyzing multimodal DESI-MRM mass spectrometry imaging data integrated with other spatial modalities including histochemistry, spatial transcriptomics, and spatial proteomics (from `.zarr` SpatialData objects).

Focus on detecting regional metabolic changes and quantifying spatial correlations across tissue sections — enabling joint analysis of metabolite distributions alongside cellular and molecular spatial context.

## Main components

- `desi_spatial_helpers.py` — core spatial analysis functions (ROI annotation, Moran's I, Pearson/Spearman correlations, KDTree-based pixel-to-cell mapping)
- `run_desi_from_config.py` — pipeline runner driven by a YAML config file

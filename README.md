# Unsupervised Deep Learning for Ecological Site Groups in MLRA 42B

**Team:** Joshua Propfe, Devon Sookhu, Joshua Wilson

## Overview

Ecological sites describe how soil, climate, and topography control which plant communities a piece of land can support and how it responds to disturbance. In practice, ecological site maps are made by linking ecological sites to soil survey (SSURGO) map units rather than by mapping them directly, and USDA is moving toward broader **ecological site groups**.

This project asks whether an **unsupervised convolutional autoencoder** can divide **MLRA 42B (Southern Rio Grande Rift)** into ecologically meaningful groups using only environmental raster data, and where those groups agree or disagree with mapped ecological sites.

The full problem statement, dataset description, methods, and preliminary results are in **[`report_ps4.pdf`](report_ps4.pdf)**.

## Current status

| Stage | Status |
|---|---|
| Data acquisition (PRISM, SOLUS, STEDIS, MLRA boundary) | Done |
| Clipping rasters to MLRA 42B (`01_clip_covariates.R`) | Done |
| Building the 19-layer covariate stack (`02_build_covariate_stack.R`) | Done |


## Repository structure

```
.
├── README.md
├── report_ps4.pdf                  # project report for this stage
├── .gitignore                      
├── src/                            # all code
│   ├── 01_clip_covariates.R
│   ├── 02_build_covariate_stack.R
├
│   
  ── preprocessing/              # small diagnostic outputs from building the stack
        ├── clip_log.csv
        └── covariate_manifest.csv      # list of input layers, types, and file paths
        ├── vif_dropped.csv
        ├── correlation_matrix.csv
        └── zscore_stats.csv
```

**Raster data are not included** because the files are too large for GitHub (each CONUS raster is roughly 15 GB, and the final 30 m stack is about 3 GB). The sections below explain where to get the data and how to rebuild the stack.

## Data sources

### Model inputs

| Group | Layers used | Resolution | Source |
|---|---|---|---|
| **Climate** | Mean annual precipitation, maximum temperature, mean temperature, minimum temperature, clear-sky solar radiation (1991–2020 normals) | 800 m | **PRISM Climate Group, Oregon State University:** https://prism.oregonstate.edu/normals/ |
| **Soil properties** | CaCO₃ (30 and 150 cm), clay (15 and 30 cm), rock fragment volume (0 and 30 cm), depth to lithic contact | 100 m | **SOLUS100** (Soil Landscapes of the United States), USDA-NRCS: https://www.nrcs.usda.gov/resources/data-and-reports/soil-landscapes-of-the-united-states-solus — also available in Google Earth Engine as `USDA/SOLUS100/V0`: https://developers.google.com/earth-engine/datasets/catalog/USDA_SOLUS100_V0 |
| **Terrain** | Slope, aspect, percentile elevation, relative mean elevation, mean elevation, topographic position index, SAGA wetness index, vertical distance to channel network | 30 m | **STEDIS** (unpublished; see note below) |
| **Spectral** | Landsat band 5 (change-free composite), NDVI interquartile range, NDVI trend since 2000 | 30 m | **STEDIS** (unpublished; see note below) |

**Note on STEDIS:** The terrain and spectral layers come from **STEDIS, an unpublished dataset** of standardized terrain and spectral derivatives for the conterminous United States, derived from public data (USGS 3DEP elevation and Landsat imagery). These layers are not publicly available yet, so they cannot be downloaded from a public source. 


### Study area boundary

| Data | Source |
|---|---|
| MLRA 42B boundary | **USDA-NRCS 2022 MLRA Geographic Database (v5.2):** https://www.nrcs.usda.gov/resources/data-and-reports/major-land-resource-area-mlra |

### Validation data (used in later stages, not model inputs)

| Data | Source |
|---|---|
| Soil survey map units and components (SSURGO) | **Web Soil Survey:** https://websoilsurvey.nrcs.usda.gov/ |
| Ecological Site Descriptions | **EDIT (Ecosystem Dynamics Interpretive Tool):** https://edit.jornada.nmsu.edu/ |

## The covariate stack

Of 23 candidate layers, **19 were retained** after a variance inflation factor (VIF) screen removed near-duplicates:
- mean temperature (VIF ≈ 268,000; PRISM computes it from maximum and minimum temperature)
- mean elevation (81.3)
- clay at 15 cm (12.4)
- minimum temperature (6.2)

| Group | Layers in final stack |
|---|---|
| Terrain (7) | `sl_32`, `aspct_32`, `perctelev_32`, `relmeanelev_32`, `tpi_32`, `swi_10`, `vdcn_5` |
| Spectral (3) | `US_6Band_ChangeFree_mosaic_B5`, `NDVI_90_IQR`, `NDVI_Trend_post2000_linearFit` |
| Climate (3) | `prism_ppt`, `prism_tmax`, `prism_solclear` |
| Soil (6) | `anylithicdpt_cm_p`, `caco3_30_cm_p`, `caco3_150_cm_p`, `claytotal_30_cm_p`, `fragvol_0_cm_p`, `fragvol_30_cm_h` |

All layers are aligned to one 30 m grid matching the STEDIS terrain grid, limited to MLRA 42B plus a 5 km buffer, and converted to z-scores. The report describes each layer and why it is used. Known issues with the current stack, which `02_build_covariate_stack_v2.R` fixes, are also listed there:
- aspect in raw degrees
- the `_h` rock-fragment layer
- near-constant clear-sky radiation
- skewed layers

## Software requirements

**R** (4.3 or later recommended)
- `terra`


## How to run

Each script has clearly marked **`>>> EDIT <<<`** sections at the top for file paths and options. Set `project_dir` to the folder containing the downloaded data, and use forward slashes (`/`) in paths, even on Windows.

### Step 1: Clip rasters to MLRA 42B (R)
```
src/01_clip_covariates.R
```
- **Inputs:** CONUS-wide rasters in a `Raw/` folder (one subfolder per covariate is fine), and the MLRA shapefile.
- **What it does:** selects MLRA 42B, buffers it by 5 km, and crops and masks each raster in its native projection. It skips finished layers on rerun and logs failures without stopping.
- **Outputs:** `out/clipped/*_42B.tif`, `out/clipped/clip_log.csv`, and the boundary files in `aoi/`.

### Step 2: Build the covariate stack (R)
```
src/02_build_covariate_stack_v2.R
```
- **First run:** creates or updates `covariate_manifest.csv` and stops. Review the `type`, `group`, and `transform` columns in Excel, save, **close Excel**, and run again.
- **What it does:** aligns all layers to the 30 m grid, applies transforms (aspect to northness and eastness, log10 for skewed layers), removes cells with missing data, runs the VIF screen (key layers can be protected), converts layers to z-scores, and writes the stack in a single pass.
- **Outputs (in `out/stack/`):** `covariate_stack_42B.tif`, `channel_order.csv`, `zscore_stats.csv`, `aoi_mask_42B.tif`, plus diagnostics (`vif_dropped.csv`, `vif_final.csv`, `correlation_matrix.csv`, `na_report.csv`, `preview_all_channels.png`).
- `02_build_covariate_stack.R` is the original version used to build the current 19-layer stack. It is kept for reproducibility.



### Typical run times
These depend heavily on disk speed and hardware. Keep all working files on an internal SSD.
- **Clipping:** minutes per layer.
- **Stack building:** roughly 30 minutes to an hour for the full 30 m extent.

## Acknowledgments

Code development, troubleshooting, and parts of the documentation were assisted by Claude (Anthropic), an AI assistant. All scripts were reviewed, run, and tested by the team.

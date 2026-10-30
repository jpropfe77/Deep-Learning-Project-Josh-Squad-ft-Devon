# Unsupervised Deep Learning for Ecological Site Groups in MLRA 42B

Group members: Joshua Propfe, Devon Sookhu, Joshua Wilson

A convolutional autoencoder compresses 32 × 32-pixel patches of a 20-channel
environmental covariate stack into 16-number fingerprints, which are clustered
with k-means into candidate ecological site groups. See `report_ps5.pdf` for
the full description and current results.

## Repository contents

| Path | What it is |
| --- | --- |
| `src/01_clip_covariates.R` | Clips CONUS-wide rasters to MLRA 42B + 5 km buffer |
| `src/02_build_covariate_stack.R` | Aligns, transforms, screens (VIF), and z-scores layers into one stack |
| `src/03_autoencoder_first_test.py` | Autoencoder training, clustering, baseline, and exports (one file) |
| `diagnostics/` | Manifest, VIF results, correlation matrix, z-score statistics, channel order |
| `results/` | Outputs from the first run (log, metrics, plots) |
| `report_ps5.pdf` | Stage 5 report |

<!-- EDIT: make the table match your repo's actual folders. -->

## Data

The rasters are too large for the repository. Sources:

- **PRISM** 1991–2020 30-year normals (precipitation, temperature, clear-sky radiation): https://prism.oregonstate.edu
- **SOLUS** soil property predictions (Nauman et al., 2024): https://doi.org/10.1002/saj2.20769
- **STEDIS** terrain and spectral derivatives (not yet published; contact the authors)
- **NRCS MLRA boundaries**: https://www.nrcs.usda.gov/resources/data-and-reports/major-land-resource-area-mlra

<!-- EDIT: confirm each link and add the STEDIS citation. -->

## Requirements

- **R** (≥ 4.2) with the `terra` package: `install.packages("terra")`
- **Python** ≥ 3.9. The Python script installs its own packages (PyTorch,
  rasterio, scikit-learn, geopandas, matplotlib, pandas, numpy) into a
  `pylibs` folder on first run, so no manual install is needed.
- **Hardware:** an NVIDIA GPU is recommended (a full run took ~6 minutes on an
  H100). CPU mode works on a 16 GB PC but training takes roughly 1–6 hours.
- **Disk:** ~3 GB for the stack, ~5–7 GB for GPU PyTorch (~1–2 GB for CPU-only).



## How to run

### Step 1–2: Build the covariate stack (R)

Edit the settings section at the top of each script (input and output paths),
then run from the repository root:

```bash
Rscript src/01_clip_covariates.R
Rscript src/02_build_covariate_stack.R
```

This produces `covariate_stack_42b.tif` (20 bands), `aoi_mask_42b.tif`,
`channel_order.csv`, and `zscore_stats.csv`.

### Step 3: Train the autoencoder and cluster (Python)

Put the four files from step 2 in one folder (the data folder), then run:

```bash
# GPU (default)
python3 src/03_autoencoder_first_test.py --base_dir /path/to/data_folder

# CPU 
python3 src/03_autoencoder_first_test.py --base_dir /path/to/data_folder --device cpu
```

On Windows, use `python` instead of `python3`.

To keep a long run going after you log out of a remote server:

```bash
nohup python3 src/03_autoencoder_first_test.py --base_dir /path/to/data_folder > run.out 2>&1 &
tail -f run.out
```

Useful options (any setting in the script's `SETTINGS` block can be overridden):

| Option | Default | Meaning |
| --- | --- | --- |
| `--base_dir` | folder containing the script | Folder with the input files; outputs are written here too |
| `--device` | `cuda` | `cuda`, `cpu`, or `auto` |
| `--n_patches` | `10000` | Number of 32 × 32 patches sampled |
| `--k_min`, `--k_max` | `3`, `12` | Range of cluster counts tested |
| `--resume` | `false` | `true` continues an interrupted training run |
| `--lib_dir` | `<base_dir>/pylibs` | Where Python packages are installed |
| `--install_only` | `false` | `true` installs packages and exits |

### Outputs

Written to `<base_dir>/outputs/ae_first_test/`:

| File | Contents |
| --- | --- |
| `run_log.txt`, `run_config.json` | Full log and every setting used |
| `completeness_report.json`, `missing_by_channel_in_aoi.csv` | Data completeness |
| `complete_patch_centers.tif` | Where complete patches can be centered (open in GIS) |
| `training_history.csv`, `loss_curves.png` | Training and validation loss |
| `per_channel_val_r2.csv`, `per_channel_val_r2.png` | Reconstruction R² per channel |
| `reconstruction_examples.png` | Original vs. rebuilt patches |
| `kmeans_metrics.csv`, `kmeans_selection.png` | Elbow, silhouette, stability, baseline agreement |
| `fingerprints.npz` | 16-number fingerprints and patch locations |
| `patch_centers_clusters.gpkg` | Patch centers with group labels for every k (open in GIS) |
| `model_best.pt` | Trained model weights |

## Reproducibility

A fixed seed (42) makes patch sampling and the train/validation split
repeatable. GPU training is not bit-for-bit deterministic, so retrained models
may differ slightly.

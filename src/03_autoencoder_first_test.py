#!/usr/bin/env python3
"""
03_autoencoder_first_test.py  (single file: installer + model + clustering)

All files are read from, and written to, the project folder on simurgh:
    /home/gojironobackup/jpropfe/ae42b      (change BASE_DIR below if it moves)

  That folder must contain:
    covariate_stack_42b.tif, aoi_mask_42b.tif, channel_order.csv, zscore_stats.csv
  Python packages install into BASE_DIR/pylibs and results go to
  BASE_DIR/outputs/ae_first_test. NOTHING in that folder is backed up:
  copy results to your home folder or PC after each run.

HOW TO RUN ON SIMURGH (GPU)
  1. Check the GPU is free:   nvidia-smi
  2. Start the run:
       cd /home/gojironobackup/jpropfe/ae42b
       nohup python3 03_autoencoder_first_test.py > run.out 2>&1 &
  3. Watch progress:   tail -f run.out      (Ctrl+C stops watching, not the run)
     Stop the run:     pkill -f 03_autoencoder_first_test.py
  4. Back up results:  cp -r outputs ~/ae42b_results_$(date +%F)

  If interrupted, rerun with  --resume true  to continue training.
  To run on CPU instead (e.g., on a PC):  --device cpu  and edit BASE_DIR.

Pipeline
  1. Scan the z-scored covariate stack band by band; read the AOI mask.
  2. Zero-fill channels where missing = 0 (caco3_150); report completeness.
  3. Sample complete 32x32 patches whose center is inside the AOI and whose
     footprint lies entirely within one 128x128 spatial block.
  4. Split by block (20% of blocks held out for validation).
  5. Train the convolutional autoencoder (MSE, Adam, LR halving, early stopping).
  6. Diagnostics: loss curves, per-channel validation R2, example reconstructions.
  7. Encode all patches to 16-number fingerprints; k-means for k = K_MIN..K_MAX
     with silhouette, inertia (elbow), and bootstrap stability.
  8. Baseline: PCA + k-means on center-pixel values; ARI vs autoencoder groups.
  9. Export patch centers with cluster labels as a GeoPackage.

Options (add after the script name)
  --n_patches 20000    override any scalar in SETTINGS
  --resume true        continue an interrupted run from its last checkpoint
"""

import argparse
import importlib
import importlib.util
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# Package installer (runs before numpy/pandas/torch are imported)
# =============================================================================
# ---- Project folder: data in, results out, packages installed here ----
BASE_DIR = Path("/home/gojironobackup/jpropfe/ae42b")


def _argv_value(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


LIB_DIR = Path(_argv_value("--lib_dir") or (BASE_DIR / "pylibs"))
DEVICE_CHOICE = (_argv_value("--device") or "cuda").lower()
REQUIRED = {  # import name -> pip name
    "numpy": "numpy", "pandas": "pandas", "rasterio": "rasterio",
    "sklearn": "scikit-learn", "geopandas": "geopandas", "pyproj": "pyproj",
    "shapely": "shapely", "matplotlib": "matplotlib", "torch": "torch",
}


def _torch_index_url():
    """Pick which PyTorch build to download."""
    system = platform.system()
    if DEVICE_CHOICE == "cpu":
        # Linux/Windows: the CPU-only build is far smaller than the default.
        # macOS: the default PyPI build is already CPU (no CUDA).
        return None if system == "Darwin" else "https://download.pytorch.org/whl/cpu"
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not m:
        return None
    v = int(m.group(1)) * 100 + int(m.group(2))
    for thr, tag in ((1208, "cu128"), (1206, "cu126"), (1204, "cu124"),
                     (1201, "cu121"), (1108, "cu118")):
        if v >= thr:
            return f"https://download.pytorch.org/whl/{tag}"
    return None


def ensure_packages():
    if sys.version_info < (3, 9):
        sys.exit(f"Python >= 3.9 required; found {sys.version.split()[0]}.")
    if not BASE_DIR.is_dir():
        sys.exit(f"Project folder not found: {BASE_DIR}\n"
                 "  Create it and put the data files there, or edit BASE_DIR.")
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(LIB_DIR))
    missing = [pip for mod, pip in REQUIRED.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"[setup] Installing into {LIB_DIR}: {', '.join(missing)}", flush=True)
    base = [sys.executable, "-m", "pip", "install", "--no-cache-dir",
            "--target", str(LIB_DIR)]
    try:
        if "torch" in missing:
            url = _torch_index_url()
            cmd = base + ["torch"] + (["--index-url", url] if url else [])
            print(f"[setup] PyTorch from: {url or 'PyPI default'}", flush=True)
            subprocess.run(cmd, check=True)
            missing.remove("torch")
        if missing:
            subprocess.run(base + missing, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"[setup] pip install failed ({e}).\n"
                 "  Scroll up for the cause. Common ones: 'Disk quota exceeded' "
                 "(use --lib_dir on a disk with more space), or no internet access.")
    except FileNotFoundError:
        sys.exit("[setup] pip not found. Try: python -m ensurepip --user")
    importlib.invalidate_caches()


ensure_packages()
if "--install_only" in sys.argv:
    i = sys.argv.index("--install_only")
    if i + 1 < len(sys.argv) and sys.argv[i + 1].lower() in ("1", "true", "yes"):
        print("[setup] Packages installed; exiting (--install_only).")
        sys.exit(0)

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

# =============================================================================
# SETTINGS  (edit here, or override any scalar from the command line)
# =============================================================================
SETTINGS = dict(
    # --- files ---
    stack_path=str(BASE_DIR / "covariate_stack_42b.tif"),
    mask_path=str(BASE_DIR / "aoi_mask_42b.tif"),          # 1 = inside MLRA 42B, 0 = buffer
    channel_order_path=str(BASE_DIR / "channel_order.csv"),
    zscore_stats_path=str(BASE_DIR / "zscore_stats.csv"),
    # channels where missing = true zero (filled with the z-score of 0)
    zero_fill_channels=["caco3_150"],
    out_dir=str(BASE_DIR / "outputs" / "ae_first_test"),
    # --- patches & split ---
    patch_size=32,
    n_patches=10000,
    block_size=128,
    val_frac=0.2,
    # --- model & training ---
    latent_dim=16,
    batch_size=128,
    lr=1e-3,
    max_epochs=200,
    early_stop_patience=8,
    lr_patience=3,          # epochs without val improvement before LR is halved
    lr_factor=0.5,
    num_workers=0,          # data-loader processes; 0 is best on a PC
    resume=False,
    # --- clustering ---
    k_min=3,
    k_max=12,
    kmeans_n_init=10,
    n_bootstrap=20,
    silhouette_sample=5000,
    standardize_latent=True,  # z-score fingerprint dims before k-means
    pca_var=0.90,             # baseline: keep PCs explaining this variance
    # --- misc ---
    seed=42,
    device="cuda",           # cuda | cpu | auto
    cpu_threads=0,          # 0 = use all CPU cores
    install_only=False,
    lib_dir="",               # package install folder (see installer)
    n_example_patches=4,
    example_channels=["sl_32", "caco3_30", "prism_ppt", "NDVI_90_IQR"],  # substrings
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in SETTINGS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"),
                           default=v)
        elif isinstance(v, (int, float, str)):
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = vars(p.parse_args())
    cfg = dict(SETTINGS)
    cfg.update(args)
    return cfg


# =============================================================================
# Utilities
# =============================================================================
def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.FileHandler(out_dir / "run_log.txt"),
                                  logging.StreamHandler(sys.stdout)])
    return logging.getLogger("ae42b")


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data loading (memory-light: never holds the whole stack)
# =============================================================================
def _to_nan(a, nodata):
    if nodata is not None and not np.isnan(nodata):
        a[a == nodata] = np.nan
    return a


def stack_info(path, log):
    import rasterio
    with rasterio.open(path) as src:
        info = dict(C=src.count, H=src.height, W=src.width, nodata=src.nodata,
                    transform=src.transform, crs=src.crs,
                    descriptions=list(src.descriptions),
                    compression=str(src.compression), block=src.block_shapes[0])
    C, H, W = info["C"], info["H"], info["W"]
    log.info(f"Stack: {C} bands, {H:,} rows x {W:,} cols, nodata={info['nodata']}, "
             f"compression={info['compression']}, block={info['block']}")
    gb_band = H * W * 4 / 1e9
    log.info(f"One band in memory: {gb_band:.2f} GB (whole stack would be "
             f"{C*gb_band:.1f} GB; it is NOT loaded whole). "
             f"Estimated peak RAM for scanning: ~{5.5*gb_band:.1f} GB")
    return info


def load_mask(path, info, log):
    import rasterio
    with rasterio.open(path) as src:
        if (src.height, src.width) != (info["H"], info["W"]):
            raise ValueError(f"Mask shape {(src.height, src.width)} != stack shape "
                             f"{(info['H'], info['W'])}")
        if not np.allclose(tuple(src.transform)[:6], tuple(info["transform"])[:6]):
            raise ValueError("Mask transform does not match the stack transform.")
        m = _to_nan(src.read(1, out_dtype="float32"), src.nodata)
    aoi = np.isfinite(m) & (m == 1)
    log.info(f"AOI cells (mask == 1): {aoi.sum():,} of {aoi.size:,}")
    return aoi


def load_channel_names(path, n_bands, descriptions, log):
    p = Path(path)
    if p.exists():
        names = pd.read_csv(p).sort_values("channel")["name"].tolist()
        if len(names) == n_bands:
            return names
        log.warning(f"channel_order has {len(names)} rows but stack has {n_bands} bands.")
    else:
        log.warning(f"{path} not found.")
    log.warning("Falling back to band descriptions for channel names.")
    return [d if d else f"band_{i+1}" for i, d in enumerate(descriptions)]


def zero_fill_values(names, cfg, log):
    """{band index: z-score of raw 0} for channels where missing means 0."""
    targets = [i for i, n in enumerate(names)
               if any(sub in n for sub in cfg["zero_fill_channels"])]
    if not targets:
        return {}
    stats = pd.read_csv(cfg["zscore_stats_path"]).set_index("name")
    out = {}
    for t in targets:
        n = names[t]
        if n not in stats.index:
            log.warning(f"{n} not in zscore_stats; zero-fill skipped."); continue
        mean, sd, tf = stats.loc[n, ["mean", "sd", "transform"]]
        if tf != "none":
            log.warning(f"{n} has transform '{tf}'; 0 is undefined. Skipped."); continue
        out[t] = float((0.0 - mean) / sd)
        log.info(f"Zero-fill {n}: missing -> z = {out[t]:.3f} (raw value 0)")
    return out


def scan_stack(info, path, aoi, names, fills, out_dir, log):
    """Read one band at a time; return a 2-D 'valid' map (all channels usable
    after zero-fill) and write the per-channel missing report."""
    import rasterio
    C, H, W = info["C"], info["H"], info["W"]
    valid = np.ones((H, W), dtype=bool)
    target_nan, rows = {}, []
    with rasterio.Env(GDAL_CACHEMAX=256), rasterio.open(path) as src:
        for b in range(C):
            t0 = time.time()
            band = _to_nan(src.read(b + 1, out_dtype="float32"), info["nodata"])
            fin = np.isfinite(band)
            del band
            rows.append(dict(name=names[b],
                             pct_missing_in_aoi=round(100 * (1 - fin[aoi].mean()), 3)))
            if b in fills:
                target_nan[b] = ~fin
            else:
                valid &= fin
            del fin
            log.info(f"  scanned band {b+1}/{C} ({names[b]}) in {time.time()-t0:.1f}s")
    pd.DataFrame(rows).to_csv(out_dir / "missing_by_channel_in_aoi.csv", index=False)

    for t, nan_map in target_nan.items():
        n_nan = int((nan_map & aoi).sum())
        n_fill = int((nan_map & valid & aoi).sum())
        log.info(f"Zero-fill {names[t]}: {n_fill:,} of {n_nan:,} missing AOI cells filled.")
        if n_nan > 0 and n_fill == 0:
            log.warning(
                f"No {names[t]} cells could be filled: they are missing in ALL bands. "
                "The stack-building step set them missing everywhere. Fill this layer "
                "with 0 in 02_build_covariate_stack.R BEFORE the missing-data step, "
                "then rebuild the stack.")
    log.info(f"AOI cells usable in all {C} channels: {100*valid[aoi].mean():.2f}%")
    return valid


# =============================================================================
# Completeness report and patch sampling
# =============================================================================
def find_candidate_patches(valid, aoi, info, P, B, out_dir, log):
    """Top-left (row, col) of every eligible, complete patch."""
    import rasterio
    H, W = valid.shape
    S = np.zeros((H + 1, W + 1), dtype=np.int32)
    np.cumsum(~valid, axis=0, dtype=np.int32, out=S[1:, 1:])
    np.cumsum(S[1:, 1:], axis=1, dtype=np.int32, out=S[1:, 1:])
    n_bad = S[P:, P:] - S[:-P, P:] - S[P:, :-P] + S[:-P, :-P]
    del S

    h = P // 2
    center_in_aoi = aoi[h:H - P + 1 + h, h:W - P + 1 + h]
    rr, cc = np.arange(H - P + 1), np.arange(W - P + 1)
    block_ok = ((rr % B) <= B - P)[:, None] & ((cc % B) <= B - P)[None, :]
    eligible = center_in_aoi & block_ok
    complete = eligible & (n_bad == 0)
    del n_bad
    n_elig, n_comp = int(eligible.sum()), int(complete.sum())
    pct = 100 * n_comp / max(n_elig, 1)
    log.info(f"Eligible patch positions (center in AOI, inside one block): {n_elig:,}")
    log.info(f"  ...of which fully complete: {n_comp:,} ({pct:.2f}%)")

    # GeoTIFF: 1 = a complete patch can be centered here (check for spatial bias)
    full = np.zeros((H, W), dtype=np.uint8)
    full[h:H - P + 1 + h, h:W - P + 1 + h] = complete
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                crs=info["crs"], transform=info["transform"], nodata=255,
                compress="deflate", tiled=True)
    with rasterio.open(out_dir / "complete_patch_centers.tif", "w", **prof) as dst:
        dst.write(full, 1)
    del full

    with open(out_dir / "completeness_report.json", "w") as f:
        json.dump(dict(aoi_cells=int(aoi.sum()),
                       pct_aoi_cells_usable=round(100 * float(valid[aoi].mean()), 3),
                       eligible_patch_positions=n_elig, complete_patch_positions=n_comp,
                       pct_eligible_patches_complete=round(pct, 3)), f, indent=2)

    idx = np.flatnonzero(complete)
    ncol = complete.shape[1]
    return idx // ncol, idx % ncol


def sample_and_split(rows, cols, cfg, W, rng, log):
    N = cfg["n_patches"]
    if len(rows) < N:
        log.warning(f"Only {len(rows):,} complete patches available; using all.")
        N = len(rows)
    pick = rng.choice(len(rows), size=N, replace=False)
    r, c = rows[pick], cols[pick]

    B = cfg["block_size"]
    nbc = int(np.ceil(W / B))
    block_id = (r // B) * nbc + (c // B)
    blocks = np.unique(block_id)
    n_val_blocks = max(1, int(round(cfg["val_frac"] * len(blocks))))
    val_blocks = rng.choice(blocks, size=n_val_blocks, replace=False)
    is_val = np.isin(block_id, val_blocks)
    log.info(f"Patches: {N:,} from {len(blocks):,} blocks; "
             f"{n_val_blocks} blocks held out -> {is_val.sum():,} val patches "
             f"({100*is_val.mean():.1f}%)")
    return r, c, block_id, is_val


def extract_patches(path, info, r, c, P, fills, log):
    """Read each patch window straight from disk (in row order for speed)."""
    import rasterio
    from rasterio.windows import Window
    X = np.empty((len(r), info["C"], P, P), dtype=np.float32)
    order = np.lexsort((c, r))
    t0 = time.time()
    with rasterio.Env(GDAL_CACHEMAX=1024), rasterio.open(path) as src:
        for j, i in enumerate(order):
            w = _to_nan(src.read(window=Window(int(c[i]), int(r[i]), P, P),
                                 out_dtype="float32"), info["nodata"])
            for t, z0 in fills.items():
                w[t][~np.isfinite(w[t])] = z0
            X[i] = w
            if (j + 1) % 2000 == 0:
                log.info(f"  read {j+1:,}/{len(r):,} patches ({time.time()-t0:.0f}s)")
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values in extracted patches; check masking.")
    log.info(f"Patch array: {X.nbytes/1e9:.2f} GB")
    return X


# =============================================================================
# Model
# =============================================================================
def build_model(C, latent_dim, P):
    import torch.nn as nn

    def enc_block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True), nn.MaxPool2d(2))

    def dec_block(i, o):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                             nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True))

    class ConvAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.s = P // 8
            self.encoder = nn.Sequential(enc_block(C, 32), enc_block(32, 64),
                                         enc_block(64, 128))
            self.fc_enc = nn.Linear(128 * self.s * self.s, latent_dim)
            self.fc_dec = nn.Sequential(nn.Linear(latent_dim, 128 * self.s * self.s),
                                        nn.ReLU(inplace=True))
            self.decoder = nn.Sequential(dec_block(128, 64), dec_block(64, 32),
                                         dec_block(32, 32))
            self.out = nn.Conv2d(32, C, 3, padding=1)  # linear output (z-scores)

        def encode(self, x):
            return self.fc_enc(self.encoder(x).flatten(1))

        def decode(self, z):
            h = self.fc_dec(z).view(-1, 128, self.s, self.s)
            return self.out(self.decoder(h))

        def forward(self, x):
            return self.decode(self.encode(x))

    return ConvAE()


# =============================================================================
# Training
# =============================================================================
def train(model, X_tr, X_va, cfg, device, out_dir, log):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    pin = device.type == "cuda"
    g = torch.Generator().manual_seed(cfg["seed"])
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr)), batch_size=cfg["batch_size"],
                       shuffle=True, generator=g, num_workers=cfg["num_workers"],
                       pin_memory=pin)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va)), batch_size=cfg["batch_size"],
                       shuffle=False, num_workers=cfg["num_workers"], pin_memory=pin)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg["lr_factor"], patience=cfg["lr_patience"])
    loss_fn = torch.nn.MSELoss()

    ckpt_last, ckpt_best = out_dir / "checkpoint_last.pt", out_dir / "model_best.pt"
    history, start_epoch, best, wait = [], 0, np.inf, 0
    if cfg["resume"] and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        history, start_epoch = ck["history"], ck["epoch"] + 1
        best, wait = ck["best"], ck["wait"]
        log.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        tr_sum, tr_n = 0.0, 0
        for (xb,) in dl_tr:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tr_sum += loss.item() * len(xb); tr_n += len(xb)

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for (xb,) in dl_va:
                xb = xb.to(device, non_blocking=True)
                va_sum += loss_fn(model(xb), xb).item() * len(xb); va_n += len(xb)

        tr_loss, va_loss = tr_sum / tr_n, va_sum / va_n
        sched.step(va_loss)
        lr_now = opt.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, lr=lr_now))
        log.info(f"Epoch {epoch:3d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
                 f"lr {lr_now:.2e} | {time.time()-t0:.1f}s")

        if va_loss < best - 1e-6:
            best, wait = va_loss, 0
            torch.save(model.state_dict(), ckpt_best)
        else:
            wait += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), history=history, epoch=epoch,
                        best=best, wait=wait), ckpt_last)
        if wait >= cfg["early_stop_patience"]:
            log.info(f"Early stopping: no val improvement for {wait} epochs.")
            break

    model.load_state_dict(torch.load(ckpt_best, map_location=device))
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    return model, pd.DataFrame(history)


# =============================================================================
# Diagnostics
# =============================================================================
def predict(model, X, device, bs=512, encode_only=False):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            out.append((model.encode(xb) if encode_only else model(xb)).cpu().numpy())
    return np.concatenate(out)


def per_channel_r2(model, X_va, names, device, out_dir, log):
    C = X_va.shape[1]
    sse, s, ss, n = np.zeros(C), np.zeros(C), np.zeros(C), 0
    for i in range(0, len(X_va), 512):
        xb = X_va[i:i + 512]
        rec = predict(model, xb, device)
        sse += ((rec - xb) ** 2).sum(axis=(0, 2, 3))
        s += xb.sum(axis=(0, 2, 3)); ss += (xb ** 2).sum(axis=(0, 2, 3))
        n += xb.shape[0] * xb.shape[2] * xb.shape[3]
    sst = ss - s ** 2 / n
    r2 = 1 - sse / np.where(sst > 0, sst, np.nan)
    df = pd.DataFrame(dict(channel=names, val_r2=r2.round(4))).sort_values("val_r2")
    df.to_csv(out_dir / "per_channel_val_r2.csv", index=False)
    log.info("Per-channel validation R2 (lowest first):\n" + df.to_string(index=False))
    return df


def plot_losses(hist, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist.epoch, hist.train_loss, label="train")
    ax.plot(hist.epoch, hist.val_loss, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (z-score units)"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "loss_curves.png", dpi=150); plt.close(fig)


def plot_r2(df, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(df) + 1))
    ax.barh(df.channel, df.val_r2); ax.set_xlabel("validation R²")
    ax.axvline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(out_dir / "per_channel_val_r2.png", dpi=150)
    plt.close(fig)


def plot_examples(model, X_va, names, cfg, device, rng, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ch = []
    for sub in cfg["example_channels"]:
        hits = [i for i, n in enumerate(names) if sub in n]
        if hits: ch.append(hits[0])
    if not ch: ch = list(range(min(4, len(names))))
    k = min(cfg["n_example_patches"], len(X_va))
    sel = rng.choice(len(X_va), size=k, replace=False)
    rec = predict(model, X_va[sel], device)
    fig, axes = plt.subplots(k, 2 * len(ch), figsize=(2.2 * 2 * len(ch), 2.2 * k),
                             squeeze=False)
    for i in range(k):
        for j, c in enumerate(ch):
            o, r = X_va[sel[i], c], rec[i, c]
            vmin, vmax = min(o.min(), r.min()), max(o.max(), r.max())
            for a, img, lab in ((axes[i, 2*j], o, "orig"), (axes[i, 2*j+1], r, "rebuilt")):
                a.imshow(img, vmin=vmin, vmax=vmax, cmap="viridis"); a.axis("off")
                if i == 0: a.set_title(f"{names[c][:18]}\n{lab}", fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "reconstruction_examples.png", dpi=150)
    plt.close(fig)


# =============================================================================
# Clustering
# =============================================================================
def cluster_sweep(Z, Xc, cfg, out_dir, log):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    N = len(Z)
    Zs = StandardScaler().fit_transform(Z) if cfg["standardize_latent"] else Z
    pca = PCA(n_components=cfg["pca_var"], svd_solver="full", random_state=seed)
    Xp = pca.fit_transform(Xc)
    log.info(f"Baseline PCA: {pca.n_components_} components explain "
             f"{100*pca.explained_variance_ratio_.sum():.1f}% of variance")
    ss = min(cfg["silhouette_sample"], N)

    rows, ae_labels, base_labels = [], {}, {}
    for k in range(cfg["k_min"], cfg["k_max"] + 1):
        km = KMeans(n_clusters=k, n_init=cfg["kmeans_n_init"], random_state=seed).fit(Zs)
        lab = km.labels_
        sizes = np.bincount(lab, minlength=k)
        sil = silhouette_score(Zs, lab, sample_size=ss, random_state=seed)

        aris = []
        for b in range(cfg["n_bootstrap"]):
            bi = rng.choice(N, size=N, replace=True)
            kb = KMeans(n_clusters=k, n_init=3, random_state=seed + b + 1).fit(Zs[bi])
            aris.append(adjusted_rand_score(lab, kb.predict(Zs)))

        kmb = KMeans(n_clusters=k, n_init=cfg["kmeans_n_init"], random_state=seed).fit(Xp)
        blab = kmb.labels_
        bsil = silhouette_score(Xp, blab, sample_size=ss, random_state=seed)

        rows.append(dict(k=k, ae_inertia=km.inertia_, ae_silhouette=sil,
                         ae_stability_ari_mean=np.mean(aris),
                         ae_stability_ari_sd=np.std(aris),
                         ae_min_cluster_size=int(sizes.min()),
                         ae_max_cluster_size=int(sizes.max()),
                         base_inertia=kmb.inertia_, base_silhouette=bsil,
                         ari_ae_vs_baseline=adjusted_rand_score(lab, blab)))
        ae_labels[k], base_labels[k] = lab, blab
        log.info(f"k={k:2d} | AE sil {sil:.3f} | stability ARI {np.mean(aris):.3f}"
                 f"±{np.std(aris):.3f} | sizes {sizes.min()}-{sizes.max()} | "
                 f"base sil {bsil:.3f} | ARI(AE,base) {rows[-1]['ari_ae_vs_baseline']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "kmeans_metrics.csv", index=False)
    return df, ae_labels, base_labels


def plot_kmeans(df, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    ax[0].plot(df.k, df.ae_inertia, "o-"); ax[0].set_title("Elbow (AE inertia)")
    ax[1].plot(df.k, df.ae_silhouette, "o-", label="autoencoder")
    ax[1].plot(df.k, df.base_silhouette, "s--", label="PCA baseline")
    ax[1].set_title("Silhouette"); ax[1].legend()
    ax[2].errorbar(df.k, df.ae_stability_ari_mean, yerr=df.ae_stability_ari_sd, fmt="o-")
    ax[2].set_title("Bootstrap stability (ARI)")
    for a in ax: a.set_xlabel("k")
    fig.tight_layout(); fig.savefig(out_dir / "kmeans_selection.png", dpi=150)
    plt.close(fig)


def export_points(r, c, P, block_id, is_val, ae_labels, base_labels,
                  transform, crs, out_dir, log):
    import geopandas as gpd
    from rasterio.transform import xy
    h = P // 2
    xs, ys = xy(transform, r + h, c + h, offset="center")
    data = dict(row=r + h, col=c + h, block_id=block_id,
                split=np.where(is_val, "val", "train"))
    for k in ae_labels:
        data[f"ae_k{k}"] = ae_labels[k]
        data[f"base_k{k}"] = base_labels[k]
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(xs, ys),
                           crs=crs.to_wkt() if crs else None)
    if crs is None:
        log.warning("Stack has no CRS; GeoPackage written without one.")
    gdf.to_file(out_dir / "patch_centers_clusters.gpkg", driver="GPKG")
    log.info(f"Wrote {len(gdf):,} points to patch_centers_clusters.gpkg")


# =============================================================================
# Main
# =============================================================================
def main():
    cfg = parse_args()
    out_dir = Path(cfg["out_dir"])
    log = setup_logging(out_dir)
    import torch
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    want = cfg["device"].lower()
    if want == "cuda" and not torch.cuda.is_available():
        log.warning("--device cuda requested but no GPU is available; using CPU.")
    use_cuda = want in ("cuda", "auto") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        log.info(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        n = cfg["cpu_threads"] or os.cpu_count() or 1
        torch.set_num_threads(n)
        log.info(f"Device: CPU, {n} threads")
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    P, B = cfg["patch_size"], cfg["block_size"]
    if P % 8 or P > B:
        raise ValueError("patch_size must be divisible by 8 and <= block_size.")

    # --- data (memory-light) ---
    t0 = time.time()
    path = cfg["stack_path"]
    info = stack_info(path, log)
    C, H, W = info["C"], info["H"], info["W"]
    names = load_channel_names(cfg["channel_order_path"], C, info["descriptions"], log)
    aoi = load_mask(cfg["mask_path"], info, log)
    fills = zero_fill_values(names, cfg, log)
    valid = scan_stack(info, path, aoi, names, fills, out_dir, log)
    rows, cols = find_candidate_patches(valid, aoi, info, P, B, out_dir, log)
    del valid, aoi
    if len(rows) == 0:
        raise RuntimeError("No complete patches found. See completeness_report.json.")
    r, c, block_id, is_val = sample_and_split(rows, cols, cfg, W, rng, log)
    del rows, cols
    X = extract_patches(path, info, r, c, P, fills, log)
    X_tr, X_va = X[~is_val], X[is_val]
    log.info(f"Data preparation took {(time.time()-t0)/60:.1f} min")

    # --- model ---
    model = build_model(C, cfg["latent_dim"], P).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model, hist = train(model, X_tr, X_va, cfg, device, out_dir, log)

    # --- diagnostics ---
    plot_losses(hist, out_dir)
    r2 = per_channel_r2(model, X_va, names, device, out_dir, log)
    plot_r2(r2, out_dir)
    plot_examples(model, X_va, names, cfg, device, rng, out_dir)

    # --- fingerprints & clustering ---
    Z = predict(model, X, device, encode_only=True)
    Xc = X[:, :, P // 2, P // 2]
    np.savez_compressed(out_dir / "fingerprints.npz", Z=Z, center_values=Xc,
                        row=r + P // 2, col=c + P // 2, is_val=is_val,
                        block_id=block_id, channel_names=np.array(names))
    df, ae_labels, base_labels = cluster_sweep(Z, Xc, cfg, out_dir, log)
    plot_kmeans(df, out_dir)
    export_points(r, c, P, block_id, is_val, ae_labels, base_labels,
                  info["transform"], info["crs"], out_dir, log)
    log.info(f"Done. Total time {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

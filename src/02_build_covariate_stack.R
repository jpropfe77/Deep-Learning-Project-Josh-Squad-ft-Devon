# =============================================================================
# 02_build_covariate_stack_v2.R
#
# STEP 2 (version 2): Turn the clipped rasters from 01_clip_covariates.R into
# ONE CNN-ready covariate stack for MLRA 42B.
#
#
#
# HOW TO RUN:
#   Run the whole script (Source, Ctrl+Shift+S). If your manifest has no
#   "transform" column yet, the script adds one with suggestions, saves the
#   manifest, and STOPS. Review it in Excel (close Excel afterwards), then run
#   the script again.
#
# Look for the  >>> EDIT <<<  banners. Nothing else needs changing.
# =============================================================================

library(terra)


# #############################################################################
# >>> EDIT 1: FOLDERS AND FILES <<<
# Use forward slashes "/" in paths, even on Windows.
# #############################################################################
project_dir  <- "C:/project"                     # <-- your working directory
aoi_file     <- "aoi/mlra42B.gpkg"               # unbuffered MLRA (from script 01)
aoi_buf_file <- "aoi/mlra42B_buffer5000m.gpkg"   # <-- buffered MLRA (from script 01)
manifest     <- "covariate_manifest.csv"         # your edited manifest
out_dir      <- "out/stack"                      # where the final stack goes
tmp_dir      <- "C:/terra_tmp"                   # <-- internal SSD temp folder


# #############################################################################
# >>> EDIT 2: TARGET GRID <<<
# ref_layer: a terrain layer whose grid is copied exactly (it is not resampled).
# target_crs below is taken from that layer automatically, so terrain is
# never reprojected. Name must match the manifest "name" column.
# #############################################################################
ref_layer  <- "sl_32_42B"          # <-- your main terrain layer
target_res <- 30                   # <-- pixel size in meters


# #############################################################################
# >>> EDIT 3: WHICH LAYERS TO USE <<<
# exclude_groups : drop every layer whose manifest "group" is listed,
#                  e.g. c("spectral"). Use c() to drop nothing.
# exclude_layers : drop individual layers by manifest name.
# keep_layers    : layers VIF may NEVER remove. Use names AFTER transform
#                  (aspect becomes <name>_northness and <name>_eastness).
# #############################################################################
exclude_groups <- c()                                   # <-- e.g. c("spectral")
exclude_layers <- c()                                   # <-- e.g. c("tpi_32_42B")
keep_layers    <- c("sl_32_42B",                        # <-- slope
                    "prism_tmax_us_30s_2020_avg_30y_42B")   # <-- one temperature layer


# #############################################################################
# >>> EDIT 4: OPTIONS <<<
# #############################################################################
do_vif         <- TRUE    # TRUE = drop redundant continuous layers
vif_group_thr  <- 10      # tier 1: within each group
vif_global_thr <- 5       # tier 2: all layers together 
log_offset     <- 1       # log10(x + log_offset); keeps zeros finite
n_sample       <- 40000   # cells used for statistics (evenly spaced sample)
min_class_frac <- 0.001   # categorical classes rarer than this -> "other"
realign        <- FALSE   # TRUE = redo alignment of every layer from scratch
seed           <- 42


# =============================================================================

# =============================================================================

dir.create(file.path(out_dir, "aligned"), recursive = TRUE, showWarnings = FALSE)
dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
terraOptions(tempdir = tmp_dir, progress = 3, memfrac = 0.6)
gd <- c("COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER")
t_start <- Sys.time()
step <- function(...) message(sprintf("\n[%s] ", format(Sys.time(), "%H:%M:%S")), ...)

## ---- 1. Read and check the manifest -----------------------------------------
step("Reading manifest")
if (!file.exists(manifest)) stop("Manifest not found: ", manifest,
                                 ". Run the original step 2 script once to create it.")
man <- read.csv(manifest, stringsAsFactors = FALSE, na.strings = c("", "NA"))
for (col in c("group", "nodata", "note", "transform")) if (is.null(man[[col]])) man[[col]] <- NA
man$type <- trimws(tolower(man$type))
man$group[is.na(man$group) | man$group == ""] <- "ungrouped"

if (!all(man$type %in% c("continuous", "categorical")))
  stop("'type' must be continuous or categorical. Problem rows: ",
       paste(man$name[!man$type %in% c("continuous", "categorical")], collapse = ", "))
if (anyDuplicated(man$name)) stop("Duplicate names: ", paste(man$name[duplicated(man$name)], collapse = ", "))
if (any(!file.exists(man$path))) stop("Missing files:\n", paste(man$path[!file.exists(man$path)], collapse = "\n"))

# Quick look at each layer's distribution (small regular sample of the clipped file)
quick_stats <- function(f) {
  v <- spatSample(rast(f)[[1]], 5000, method = "regular", na.rm = TRUE)[, 1]
  v <- v[is.finite(v)]
  if (length(v) < 10) return(c(min = NA, skew = NA, cv = NA, n_unique = NA))
  s <- sd(v)
  c(min = min(v), skew = if (s > 0) mean((v - mean(v))^3) / s^3 else 0,
    cv = if (mean(v) != 0) s / abs(mean(v)) else NA, n_unique = length(unique(v)))
}

# If there is no transform column yet: add one with suggestions and stop for review
if (all(is.na(man$transform))) {
  step("Adding a 'transform' column with suggestions")
  qs <- t(sapply(man$path, quick_stats))
  is_aspect <- grepl("asp", man$name, ignore.case = TRUE) & man$type == "continuous"
  is_skewed <- man$type == "continuous" & !is_aspect & qs[, "skew"] > 1.5 & qs[, "min"] >= 0
  man$transform <- ifelse(is_aspect, "aspect", ifelse(is_skewed %in% TRUE, "log10", "none"))
  man$note <- ifelse(is_aspect, "aspect -> northness + eastness",
              ifelse(is_skewed %in% TRUE, sprintf("skewness %.1f -> log10 suggested", qs[, "skew"]),
                     ifelse(is.na(man$note), "", man$note)))
  write.csv(man, manifest, row.names = FALSE, na = "")
  print(man[, c("name", "type", "group", "transform")], row.names = FALSE)
  stop("\n\nAdded a 'transform' column to ", manifest, " (shown above).\n",
       "Review it in Excel: values are none, log10, or aspect.\n",
       "Save as CSV, CLOSE Excel, then run this script again.", call. = FALSE)
}
man$transform <- trimws(tolower(ifelse(is.na(man$transform), "none", man$transform)))
if (!all(man$transform %in% c("none", "log10", "aspect")))
  stop("'transform' must be none, log10, or aspect. Problem rows: ",
       paste(man$name[!man$transform %in% c("none", "log10", "aspect")], collapse = ", "))

# Apply exclusions
drop <- man$group %in% exclude_groups | man$name %in% exclude_layers
if (any(drop)) message("Excluded: ", paste(man$name[drop], collapse = ", "))
bad_ex <- setdiff(exclude_layers, man$name)
if (length(bad_ex)) warning("exclude_layers not in manifest: ", paste(bad_ex, collapse = ", "), call. = FALSE)
man <- man[!drop, ]
message(nrow(man), " layers in use (", sum(man$type == "continuous"), " continuous, ",
        sum(man$type == "categorical"), " categorical)")

## ---- 2. Warnings about specific layers --------------------------------------
pi_like <- grepl("_(h|l)(_|$)", man$name) & grepl("cm", man$name)
if (any(pi_like))
  warning("These look like SOLUS prediction-INTERVAL layers (_h = high, _l = low), not predictions (_p):\n  ",
          paste(man$name[pi_like], collapse = "\n  "),
          "\nCheck the SOLUS documentation and swap in the _p files if so.", call. = FALSE)

step("Checking layer distributions")
qs <- t(sapply(man$path, quick_stats)); rownames(qs) <- man$name
lowvar <- man$type == "continuous" & qs[, "cv"] < 0.02
if (any(lowvar %in% TRUE))
  warning("Near-constant layers (variation < 2% of the mean). Z-scoring will exaggerate them:\n  ",
          paste(sprintf("%s (CV %.3f)", man$name[lowvar %in% TRUE], qs[lowvar %in% TRUE, "cv"]), collapse = "\n  "),
          "\nConsider replacing (e.g. PRISM solclear -> solslope) or excluding.", call. = FALSE)
skew_left <- man$type == "continuous" & man$transform == "none" & qs[, "skew"] > 2
if (any(skew_left %in% TRUE))
  message("NOTE: strongly skewed layers with no transform (consider log10): ",
          paste(man$name[skew_left %in% TRUE], collapse = ", "))
cat_numeric <- man$type == "categorical" & qs[, "n_unique"] > 30
if (any(cat_numeric %in% TRUE))
  warning("Marked categorical but have >30 distinct values (probably continuous): ",
          paste(man$name[cat_numeric %in% TRUE], collapse = ", "), call. = FALSE)
bad_log <- man$transform == "log10" & qs[, "min"] <= -log_offset
if (any(bad_log %in% TRUE))
  stop("log10 needs values > -log_offset. These have smaller values: ",
       paste(man$name[bad_log %in% TRUE], collapse = ", "))

## ---- 3. Shared grid (copied from the reference terrain layer) ---------------
step("Building the shared grid")
aoi_buf_v <- vect(aoi_buf_file)
if (!is.null(ref_layer) && ref_layer %in% man$name) {
  ref <- rast(man$path[man$name == ref_layer])
  if (abs(res(ref)[1] - target_res) > 1e-6)
    stop("ref_layer has ", res(ref)[1], " m pixels but target_res is ", target_res)
  target_crs <- crs(ref)
  aoi_buf <- project(aoi_buf_v, target_crs)
  tmpl <- rast(align(ext(aoi_buf), ref, snap = "out"), resolution = res(ref), crs = target_crs)
} else {
  warning("ref_layer not found in manifest; using EPSG:5070 with a snapped grid.", call. = FALSE)
  target_crs <- "EPSG:5070"
  aoi_buf <- project(aoi_buf_v, target_crs)
  e <- ext(aoi_buf); r0 <- target_res
  tmpl <- rast(ext(floor(xmin(e) / r0) * r0, ceiling(xmax(e) / r0) * r0,
                   floor(ymin(e) / r0) * r0, ceiling(ymax(e) / r0) * r0),
               resolution = r0, crs = target_crs)
}
aoi <- project(vect(aoi_file), target_crs)
message(sprintf("Grid: %d rows x %d cols at %g m (%.1f million cells)",
                nrow(tmpl), ncol(tmpl), target_res, ncell(tmpl) / 1e6))
grid_sig <- paste(nrow(tmpl), ncol(tmpl), paste(round(as.vector(ext(tmpl)), 3), collapse = ","), sep = "|")

## ---- 4. Transform + align each layer (reusing previous work) ----------------
src_cell_m <- function(r) {
  if (is.lonlat(r)) {
    cx <- mean(c(xmin(r), xmax(r))); cy <- mean(c(ymin(r), ymax(r)))
    sqrt(expanse(vect(ext(cx, cx + res(r)[1], cy, cy + res(r)[2]), crs = crs(r)), unit = "m"))
  } else mean(res(r)) * linearUnits(r)
}

reg_file <- file.path(out_dir, "aligned", "_registry.csv")
registry <- if (file.exists(reg_file)) read.csv(reg_file, colClasses = "character", na.strings = character(0)) else
  data.frame(out_name = character(), source = character(), transform = character(),
             nodata = character(), grid = character())

layers <- list()   # one entry per OUTPUT layer: name, file, type, group
align_log <- list()
for (i in seq_len(nrow(man))) {
  nm <- man$name[i]; type <- man$type[i]; tf <- man$transform[i]
  nod <- ifelse(is.na(man$nodata[i]), "", as.character(man$nodata[i]))
  out_names <- if (tf == "aspect") paste0(nm, c("_northness", "_eastness")) else nm
  out_files <- file.path(out_dir, "aligned", paste0(out_names, ".tif"))

  # Reuse if already aligned from the same source, transform, NoData and grid
  hit <- sapply(out_names, function(o) isTRUE(any(registry$out_name == o & registry$source == man$path[i] &
                                                    registry$transform == tf & registry$nodata == nod &
                                                    registry$grid == grid_sig)))
  if (!realign && all(hit) && all(file.exists(out_files))) {
    message(sprintf("[%d/%d] %s: reusing aligned file", i, nrow(man), nm))
    method <- "reused"
  } else {
    r <- rast(man$path[i])[[1]]
    if (nod != "") NAflag(r) <- as.numeric(nod)
    src_m <- src_cell_m(r)
    finer <- src_m < 0.75 * target_res
    # Transform in the source grid, BEFORE resampling
    if (tf == "log10") {
      r <- log10(r + log_offset)
    } else if (tf == "aspect") {
      flat <- r < 0 | r > 360                         # flat cells are often coded -1
      rad  <- r * pi / 180
      r <- c(ifel(flat, 0, cos(rad)), ifel(flat, 0, sin(rad)))   # northness, eastness
    }
    same_grid <- same.crs(r, tmpl) && isTRUE(all.equal(res(r), res(tmpl))) &&
                 isTRUE(all.equal(origin(r), origin(tmpl), tolerance = 1e-6))
    method <- if (same_grid) "none (same grid)" else if (type == "continuous") {
      if (finer) "average" else "bilinear"
    } else {
      if (finer) "mode" else "near"
    }
    message(sprintf("[%d/%d] %s: %.0f m -> %g m, transform = %s, method = %s",
                    i, nrow(man), nm, src_m, target_res, tf, method))
    dt <- if (type == "categorical") "INT4S" else "FLT4S"
    for (k in seq_along(out_names)) {
      rk <- r[[k]]
      if (same_grid) {
        writeRaster(extend(crop(rk, tmpl, snap = "near"), tmpl), out_files[k],
                    overwrite = TRUE, datatype = dt, gdal = gd)
      } else {
        project(rk, tmpl, method = method, filename = out_files[k], overwrite = TRUE,
                datatype = dt, gdal = gd)
      }
      registry <- registry[registry$out_name != out_names[k], ]
      registry <- rbind(registry, data.frame(out_name = out_names[k], source = man$path[i],
                                             transform = tf, nodata = nod, grid = grid_sig))
      write.csv(registry, reg_file, row.names = FALSE)
    }
  }
  for (k in seq_along(out_names))
    layers[[length(layers) + 1]] <- data.frame(name = out_names[k], file = out_files[k],
                                               type = type, group = man$group[i], transform = tf)
  align_log[[i]] <- data.frame(name = nm, transform = tf, method = method)
}
layers <- do.call(rbind, layers)
write.csv(do.call(rbind, align_log), file.path(out_dir, "alignment_log.csv"), row.names = FALSE)

stk <- rast(layers$file); names(stk) <- layers$name
buf_r <- rasterize(aoi_buf, tmpl, field = 1, background = NA)
names(buf_r) <- "inbuf"

## ---- 5. Statistics sample (complete cells inside the buffer only) -----------
step("Sampling cells for statistics (evenly spaced)")
set.seed(seed)
# Oversample, then keep only complete cells (na.rm alone is unreliable in some terra versions)
samp_all <- spatSample(c(stk, buf_r), size = n_sample * 3, method = "regular", as.df = TRUE)
samp_all <- samp_all[rowSums(!is.finite(as.matrix(samp_all))) == 0, , drop = FALSE]
if (nrow(samp_all) > n_sample) samp_all <- samp_all[sort(sample(nrow(samp_all), n_sample)), , drop = FALSE]
samp_all$inbuf <- NULL
message("Complete cells sampled: ", nrow(samp_all))
if (nrow(samp_all) < 1000) stop("Too few complete cells sampled. Check na_report / layer coverage.")

cont <- layers$name[layers$type == "continuous"]
cats <- layers$name[layers$type == "categorical"]
samp <- samp_all[, cont, drop = FALSE]

zero_var <- names(samp)[sapply(samp, sd) == 0]
if (length(zero_var)) warning("Constant layers removed: ", paste(zero_var, collapse = ", "), call. = FALSE)
cont <- setdiff(cont, zero_var); samp <- samp[, cont, drop = FALSE]

## ---- 6. VIF with protected layers --------------------------------------------
vif_calc <- function(df) {
  R <- cor(df)
  setNames(diag(solve(R + diag(1e-6, ncol(R)))), names(df))
}
vif_step <- function(df, thr, tier, protect) {
  dropped <- NULL
  repeat {
    if (ncol(df) < 2) break
    v <- vif_calc(df)
    cand <- v[!names(v) %in% protect]          # protected layers can't be removed
    if (!length(cand) || max(cand) <= thr) break
    worst <- names(which.max(cand))
    dropped <- rbind(dropped, data.frame(tier = tier, layer = worst, vif = round(max(cand), 2)))
    df <- df[, setdiff(names(df), worst), drop = FALSE]
  }
  list(keep = names(df), dropped = dropped)
}

bad_keep <- setdiff(keep_layers, layers$name)
if (length(bad_keep)) warning("keep_layers not found (check names after transform): ",
                              paste(bad_keep, collapse = ", "), call. = FALSE)
if (do_vif && length(cont) > 1) {
  step("VIF reduction")
  grp   <- layers$group[match(cont, layers$name)]
  tier1 <- lapply(split(cont, grp), function(g)
    vif_step(samp[, g, drop = FALSE], vif_group_thr, "1: within group", keep_layers))
  keep1 <- unlist(lapply(tier1, `[[`, "keep"), use.names = FALSE)
  tier2 <- vif_step(samp[, keep1, drop = FALSE], vif_global_thr, "2: all layers", keep_layers)
  cont_keep <- cont[cont %in% tier2$keep]
  vif_log <- do.call(rbind, c(lapply(tier1, `[[`, "dropped"), list(tier2$dropped)))
  if (is.null(vif_log)) vif_log <- data.frame(tier = character(), layer = character(), vif = numeric())
  write.csv(vif_log, file.path(out_dir, "vif_dropped.csv"), row.names = FALSE)
  message(sprintf("VIF kept %d of %d continuous layers. Dropped: %s", length(cont_keep), length(cont),
                  if (nrow(vif_log)) paste(vif_log$layer, collapse = ", ") else "none"))
  fv <- vif_calc(samp[, cont_keep, drop = FALSE])
  write.csv(data.frame(layer = names(fv), final_vif = round(fv, 2), protected = names(fv) %in% keep_layers),
            file.path(out_dir, "vif_final.csv"), row.names = FALSE)
} else {
  cont_keep <- cont
  message("VIF skipped; keeping all ", length(cont), " continuous layers")
}
write.csv(round(cor(samp[, cont_keep, drop = FALSE]), 3), file.path(out_dir, "correlation_matrix.csv"))

## ---- 7. Z-score stats and categorical classes --------------------------------
mu  <- sapply(samp[, cont_keep, drop = FALSE], mean)
sdv <- sapply(samp[, cont_keep, drop = FALSE], sd)
write.csv(data.frame(name = cont_keep, mean = mu, sd = sdv,
                     transform = layers$transform[match(cont_keep, layers$name)]),
          file.path(out_dir, "zscore_stats.csv"), row.names = FALSE)

cat_classes <- lapply(cats, function(nm) {
  fq <- freq(stk[[nm]])
  frac <- fq$count / sum(fq$count)
  list(keep = fq$value[frac >= min_class_frac], rare = fq$value[frac < min_class_frac])
})
names(cat_classes) <- cats
out_names <- c(cont_keep, unlist(lapply(cats, function(nm) {
  cc <- cat_classes[[nm]]
  c(paste0(nm, "_", cc$keep), if (length(cc$rare)) paste0(nm, "_other"))
})))
message(length(out_names), " output channels (", length(cont_keep), " continuous, ",
        length(out_names) - length(cont_keep), " categorical 0/1)")

## ---- 8. Build the final stack in ONE pass -------------------------------------
step("Writing the final stack (single pass; progress bar below)")
in_stk <- c(stk[[c(cont_keep, cats)]], buf_r)
final_path <- file.path(out_dir, "covariate_stack_42B.tif")
out <- rast(tmpl, nlyrs = length(out_names)); names(out) <- out_names
n_in <- nlyr(in_stk) - 1
na_cnt <- numeric(n_in); buf_cnt <- 0

readStart(in_stk)
b <- writeStart(out, final_path, overwrite = TRUE, datatype = "FLT4S", gdal = gd, names = out_names)
for (i in seq_len(b$n)) {
  v <- readValues(in_stk, row = b$row[i], nrows = b$nrows[i], col = 1, ncols = ncol(in_stk), mat = TRUE)
  inb <- !is.na(v[, n_in + 1])
  x <- v[, seq_len(n_in), drop = FALSE]
  if (any(inb)) { na_cnt <- na_cnt + colSums(is.na(x[inb, , drop = FALSE])); buf_cnt <- buf_cnt + sum(inb) }
  valid <- inb & rowSums(is.na(x)) == 0
  res_m <- matrix(NA_real_, nrow(x), length(out_names))
  if (any(valid)) {
    xv <- x[valid, , drop = FALSE]
    cols <- list()
    if (length(cont_keep)) cols[[1]] <- sweep(sweep(xv[, seq_along(cont_keep), drop = FALSE], 2, mu), 2, sdv, "/")
    for (nm in cats) {
      cv <- xv[, length(cont_keep) + match(nm, cats)]
      cc <- cat_classes[[nm]]
      oh <- sapply(cc$keep, function(k) as.numeric(cv == k))
      if (length(cc$rare)) oh <- cbind(oh, as.numeric(cv %in% cc$rare))
      cols[[length(cols) + 1]] <- matrix(oh, nrow = nrow(xv))
    }
    res_m[valid, ] <- do.call(cbind, cols)
  }
  writeValues(out, res_m, b$row[i], b$nrows[i])
}
out <- writeStop(out)
readStop(in_stk)

na_report <- data.frame(name = names(in_stk)[seq_len(n_in)],
                        pct_missing_in_aoi = round(100 * na_cnt / max(buf_cnt, 1), 2))
write.csv(na_report, file.path(out_dir, "na_report.csv"), row.names = FALSE)
if (any(na_report$pct_missing_in_aoi > min(na_report$pct_missing_in_aoi) + 5))
  warning("Some layers have >5% more missing cells than others (they punch holes in the whole stack). ",
          "See na_report.csv.", call. = FALSE)

## ---- 9. Channel order, AOI mask, preview --------------------------------------
write.csv(data.frame(channel = seq_along(out_names), name = out_names,
                     kind = ifelse(out_names %in% cont_keep, "continuous (z-score)", "categorical (0/1)"),
                     transform = ifelse(out_names %in% cont_keep,
                                        layers$transform[match(out_names, layers$name)], "one-hot")),
          file.path(out_dir, "channel_order.csv"), row.names = FALSE)

step("Writing AOI mask and preview")
aoi_mask <- mask(rasterize(aoi, tmpl, background = 0), out[[1]])
writeRaster(aoi_mask, file.path(out_dir, "aoi_mask_42B.tif"), overwrite = TRUE, datatype = "INT1U", gdal = gd)

n <- nlyr(out); nc <- ceiling(sqrt(n)); nr <- ceiling(n / nc)
png(file.path(out_dir, "preview_all_channels.png"), width = 400 * nc, height = 350 * nr)
plot(out, nc = nc, nr = nr, maxcell = 2e5, axes = FALSE, legend = TRUE, maxnl = n)
invisible(dev.off())

message("\n==== DONE in ", round(as.numeric(difftime(Sys.time(), t_start, units = "mins")), 1), " minutes ====")
message("Stack: ", final_path, "  (", nlyr(out), " channels)")
message("Give the model: covariate_stack_42B.tif, channel_order.csv, zscore_stats.csv, aoi_mask_42B.tif")
message("Check: vif_dropped.csv, vif_final.csv, correlation_matrix.csv, na_report.csv, preview_all_channels.png")

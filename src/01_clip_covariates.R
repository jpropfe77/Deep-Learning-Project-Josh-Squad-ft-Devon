# =============================================================================
# 01_clip_covariates.R
#
# STEP 1 of 2: Clip each large raster down to a buffered MLRA 42B area.
#
# - Each raster is clipped in its OWN projection and pixel size (no resampling
#   here, so no values are altered).
# - The script LOOPS over every raster automatically.  Just put the files in place.
# =============================================================================

library(terra)


# #############################################################################
# >>> EDIT 1: FOLDER LOCATIONS <<<
# Use forward slashes "/" in paths, even on Windows.
# #############################################################################
project_dir <- "C:/project"        # <-- your working directory (main project folder)
raw_dir     <- "Raw"               # <-- folder holding the raw covariate subfolders.
                                   #     If raw files are on another drive, use a
                                   #     full path, e.g. "E:/raw"
out_dir     <- "out/clipped"       # where the clipped rasters will be written
tmp_dir     <- "C:/terra_tmp"      # <-- a folder on your INTERNAL drive with free
                                   #     space, for terra's temporary files


# #############################################################################
# >>> EDIT 2: MLRA BOUNDARY FILE <<<
# Run this once in the console to see the column names of your MLRA file:
#     names(vect("aoi/your_mlra_file.shp"))
# Then put the column that holds codes like "42B" in mlra_field.
# #############################################################################

mlra_file  <- "MLRA_42B.shp"  # <--MLRA shapefile name
mlra_field <- "MLRARSYM"            # <-- column holding MLRA codes (check!)
mlra_code  <- "42B"                 # <-- the MLRA you want


# #############################################################################
# >>> EDIT 3: BUFFER DISTANCE (meters) <<<
# Extra margin kept around the MLRA so CNN patches at the edge are complete.
# Rule of thumb: at least half your CNN patch width in meters,
#   e.g. 11 x 11 patch at 30 m  -> half width ~165 m
#        11 x 11 patch at 100 m -> half width ~550 m
# 5000 m (5 km) 
# #############################################################################
buffer_m <- 5000


# #############################################################################
# >>> 4 
# automatically finds EVERY .tif/.tiff inside raw_dir and
#                     its subfolders. 
# 
# #############################################################################


# Option A
tif_files <- list.files(raw_dir, pattern = "\\.tiff?$", recursive = TRUE,
                        full.names = TRUE, ignore.case = TRUE)

print (tif_files)
# =============================================================================

# =============================================================================

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
terraOptions(tempdir = tmp_dir, progress = 0)

## ---- Check the file list ----------------------------------------------------
if (length(tif_files) == 0) stop("No .tif files found in '", raw_dir, "'. Check raw_dir.")
missing <- tif_files[!file.exists(tif_files)]
if (length(missing)) stop("These files do not exist:\n", paste(missing, collapse = "\n"))

layer_names <- tools::file_path_sans_ext(basename(tif_files))
dups <- layer_names[duplicated(layer_names)]
if (length(dups)) stop("Two or more files share the same name (outputs would overwrite each other): ",
                       paste(unique(dups), collapse = ", "),
                       "\nRename them so every file name is unique.")
message("Found ", length(tif_files), " raster(s) to clip.")

## ---- Build the area of interest ---------------------------------------------
mlra_file <- "Raw/MLRA_42B/MLRA_42B.shp"
mlra <- vect(mlra_file)
if (!mlra_field %in% names(mlra))
  stop("Column '", mlra_field, "' not found in the MLRA file. Columns are: ",
       paste(names(mlra), collapse = ", "))

aoi <- mlra[values(mlra)[[mlra_field]] %in% mlra_code, ]
if (nrow(aoi) == 0)
  stop("No polygons with ", mlra_field, " = '", mlra_code, "'. Values present include: ",
       paste(head(sort(unique(values(mlra)[[mlra_field]])), 20), collapse = ", "))

aoi     <- aggregate(aoi)                      # merge multi-part polygons into one
aoi_m   <- project(aoi, "EPSG:5070")           # meters, so the buffer distance is exact
aoi_buf <- buffer(aoi_m, buffer_m)

# Save both: buffered (for clipping) and unbuffered (to limit training/prediction later)
dir.create("aoi", showWarnings = FALSE)
writeVector(aoi_m, file.path("aoi", paste0("mlra", mlra_code, ".gpkg")), overwrite = TRUE)
writeVector(aoi_buf, file.path("aoi", paste0("mlra", mlra_code, "_buffer", buffer_m, "m.gpkg")),
            overwrite = TRUE)
message(sprintf("AOI: MLRA %s, %.0f km2 (%.0f km2 with buffer)",
                mlra_code, expanse(aoi_m, unit = "km", transform = FALSE),
                expanse(aoi_buf, unit = "km", transform = FALSE)))

## ---- Clip function (used for every raster) ----------------------------------
# Describe a clipped file for the log
layer_info <- function(out, nm, f, status, mins) {
  rc <- rast(out)
  data.frame(layer = nm, status = status, minutes = mins,
             crs = crs(rc, describe = TRUE)$code, res_x = res(rc)[1], res_y = res(rc)[2],
             nrow = nrow(rc), ncol = ncol(rc), datatype = datatype(rc)[1],
             na_flag = NAflag(rc)[1], input = f, output = out)
}

clip_one <- function(f, nm) {
  out <- file.path(out_dir, paste0(nm, "_", mlra_code, ".tif"))
  if (file.exists(out)) {
    message("  SKIP (already done): ", nm)
    return(layer_info(out, nm, f, "skipped (exists)", 0))
  }
  t0 <- Sys.time()
  r  <- rast(f)
  if (nlyr(r) > 1) warning(nm, " has ", nlyr(r), " bands; all bands are kept.")
  if (crs(r) == "") stop("no coordinate system found. Is a .tfw/.prj/.aux.xml sidecar file missing?")

  aoi_r <- project(aoi_buf, crs(r))            # put the AOI in the raster's projection
  if (!relate(ext(r), ext(aoi_r), "intersects"))
    stop("raster does not overlap MLRA ", mlra_code, " (check its projection)")

  tmp <- paste0(out, ".partial.tif")           # write to a temp name first, so a crash
  crop(r, aoi_r, snap = "out", mask = TRUE,    # never leaves a half-written "finished" file
       filename = tmp, overwrite = TRUE,
       datatype = datatype(r)[1],
       gdal = c("COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"))
  file.rename(tmp, out)

  mins <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1)
  info <- layer_info(out, nm, f, "clipped", mins)
  message(sprintf("  DONE: %s  (%d x %d cells, %.1f MB, %.1f min)",
                  nm, info$nrow, info$ncol, file.size(out) / 1e6, mins))
  info
}

## ---- Run the loop -----------------------------------------------------------
log <- list()
for (i in seq_along(tif_files)) {
  message(sprintf("[%d/%d] %s", i, length(tif_files), layer_names[i]))
  log[[i]] <- tryCatch(
    clip_one(tif_files[i], layer_names[i]),
    error = function(e) {
      message("  FAILED: ", layer_names[i], " -> ", conditionMessage(e))
      unlink(file.path(out_dir, paste0(layer_names[i], "_", mlra_code, ".tif.partial.tif")))
      data.frame(layer = layer_names[i], status = paste("FAILED:", conditionMessage(e)), minutes = NA)
    })
}

## ---- Save a log so you can check what happened ------------------------------
all_cols <- unique(unlist(lapply(log, names)))
log_df <- do.call(rbind, lapply(log, function(d) { d[setdiff(all_cols, names(d))] <- NA; d[all_cols] }))
write.csv(log_df, file.path(out_dir, "clip_log.csv"), row.names = FALSE)

n_fail <- sum(grepl("^FAILED", log_df$status))
message("\nFinished. ", sum(log_df$status == "clipped"), " clipped, ",
        sum(grepl("^skipped", log_df$status)), " skipped, ", n_fail, " failed.")
message("Log written to ", file.path(out_dir, "clip_log.csv"))
if (n_fail) message("Fix the failed layers (see log), then simply run the script again.")

# Extract MLRA 42B (Southern Rio Grande Rift) from NRCS MLRA v5.2
# Outputs: zipped shapefile + KML, both WGS84 (EPSG:4326)

library(sf)

# ---- Paths (edit) ----
in_dir  <- "path/to/MLRA_52"           # folder containing MLRA_52.shp
out_dir <- "path/to/output"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Read ----
mlra <- st_read("MLRA_52.shp", quiet = TRUE)

# ---- Subset ----
# NOTE: filter on MLRARSYM, NOT MLRA_ID.
# MLRA_ID == 42 is actually 42A; 42B has MLRA_ID == 43.
mlra_42b <- mlra[mlra$MLRARSYM == "42B", ]
stopifnot(nrow(mlra_42b) == 1)

# ---- Checks ----
mlra_42b <- st_make_valid(mlra_42b)
mlra_42b <- st_transform(mlra_42b, 4326)   # already WGS84; explicit for KML
print(st_bbox(mlra_42b))
print(units::set_units(st_area(st_transform(mlra_42b, 5070)), km^2))  # ~41,763 km2

# Keep short field names (shapefile limit = 10 chars; these already comply)
mlra_42b <- mlra_42b[, c("MLRA_ID", "MLRARSYM", "MLRA_NAME", "LRRSYM", "LRR_NAME")]

# ---- Write shapefile ----
shp_name <- "MLRA_42B"
st_write(mlra_42b, file.path(out_dir, paste0(shp_name, ".shp")),
         delete_dsn = TRUE, quiet = TRUE)


# ---- Zip shapefile (to getwd(); flat, no subfolders inside zip) ----
# Explicit extensions so the zip doesn't pick up MLRA_42B.kml or the MLRA_52 source files
shp_exts  <- c("shp", "shx", "dbf", "prj", "cpg")
shp_files <- paste0(shp_name, ".", shp_exts)
shp_files <- shp_files[file.exists(shp_files)]
zip_path  <- paste0(shp_name, ".zip")
if (file.exists(zip_path)) file.remove(zip_path)
zip(zipfile = zip_path, files = shp_files, flags = "-j")   # -j = junk paths

# ---- Write KML ----
st_write(mlra_42b, paste0(shp_name, ".kml"),   # writes to getwd()
         driver = "KML", delete_dsn = TRUE, quiet = TRUE)

# ---- Quick visual check ----
plot(st_geometry(mlra_42b), col = "tan", main = "MLRA 42B - Southern Rio Grande Rift")

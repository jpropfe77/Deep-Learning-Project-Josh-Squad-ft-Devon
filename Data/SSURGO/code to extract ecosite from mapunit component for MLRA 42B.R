library(soilDB)
q <- "SELECT mu.mukey, c.cokey, c.compname, c.comppct_r, c.majcompflag,
             e.ecoclassid, e.ecoclassname, e.ecoclasstypename
      FROM legend l
      JOIN mapunit mu ON mu.lkey = l.lkey
      JOIN component c ON c.mukey = mu.mukey
      LEFT JOIN coecoclass e ON e.cokey = c.cokey
      WHERE l.areasymbol = 'XX###'"   # your survey area
x <- SDA_query(q)
write.csv(x, "data/raw/sda_ecosites_XX###_2026-10-01.csv", row.names = FALSE)

library(soilDB)
sa <- SDA_query("SELECT DISTINCT l.areasymbol, l.areaname, lo.areasymbol AS mlra
                 FROM legend l
                 JOIN laoverlap lo ON lo.lkey = l.lkey
                 WHERE lo.areatypename = 'MLRA' AND lo.areasymbol LIKE '42%'
                 ORDER BY l.areasymbol")
sa
q <- "SELECT l.areasymbol, mu.mukey, c.cokey, c.compname, c.comppct_r, c.majcompflag,
             e.ecoclassid, e.ecoclassname, e.ecoclasstypename
      FROM legend l
      JOIN laoverlap lo ON lo.lkey = l.lkey
           AND lo.areatypename = 'MLRA' AND lo.areasymbol = '42B'
      JOIN muaoverlap mao ON mao.lareaovkey = lo.lareaovkey
      JOIN mapunit mu ON mu.mukey = mao.mukey
      JOIN component c ON c.mukey = mu.mukey
      LEFT JOIN coecoclass e ON e.cokey = c.cokey"
x <- SDA_query(q)
nrow(x)


length(unique(x$mukey))          # number of map units
length(unique(x$cokey))          # number of components
sum(is.na(x$ecoclassid))         # components with no ecological site
table(x$ecoclasstypename, useNA = "ifany")
sum(duplicated(x$cokey))         # components with >1 ecoclass row

# How much area lacks an ecosite? Major vs minor components?
aggregate(comppct_r ~ majcompflag, data = x[is.na(x$ecoclassid), ], FUN = sum)
head(sort(table(x$compname[is.na(x$ecoclassid)]), decreasing = TRUE), 15)

# Do component percents sum to ~100 per map unit?
summary(tapply(x$comppct_r, x$mukey, sum))

# ID format (still need this one)
table(substr(x$ecoclassid, 1, 5), useNA = "ifany")

# Treat missing sites as their own class (change to drop them if you prefer)
x$site <- ifelse(is.na(x$ecoclassid), "No site", x$ecoclassid)

# Sum component % by site within each map unit
agg <- aggregate(comppct_r ~ areasymbol + mukey + site, data = x, FUN = sum)

# One row per map unit: dominant site, runner-up, and tie flag
res <- do.call(rbind, lapply(split(agg, agg$mukey), function(d) {
  d <- d[order(-d$comppct_r), ]
  data.frame(areasymbol  = d$areasymbol[1],
             mukey       = d$mukey[1],
             dom_site    = d$site[1],
             dom_pct     = d$comppct_r[1],
             second_site = if (nrow(d) > 1) d$site[2] else NA,
             second_pct  = if (nrow(d) > 1) d$comppct_r[2] else 0,
             n_sites     = nrow(d),
             tie         = nrow(d) > 1 && d$comppct_r[1] == d$comppct_r[2])
}))

# Add site names
nm <- unique(x[!is.na(x$ecoclassid), c("ecoclassid", "ecoclassname")])
res$dom_name <- nm$ecoclassname[match(res$dom_site, nm$ecoclassid)]

write.csv(res, "mu_dominant_ecosite_MLRA42B.csv", row.names = FALSE)
table(res$tie); summary(res$dom_pct)

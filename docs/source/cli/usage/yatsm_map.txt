$ yatsm map --help
Usage: yatsm map [OPTIONS] <map_type> <date> <output>

  Map types: coef, predict, class, pheno

  Map QA flags:
      - 0 => no values
      - 1 => before
      - 2 => after
      - 3 => intersect

  Examples:
  > yatsm map --coef intercept --coef slope
  ... --band 3 --band 4 --band 5 --ndv -9999
  ... coef 2000-01-01 coef_map.gtif

  > yatsm map -c intercept -c slope -b 3 -b 4 -b 5 --ndv -9999
  ... coef 2000-01-01 coef_map.gtif

  > yatsm map --date "%Y-%j" predict 2000-001 prediction.gtif

  > yatsm map --result "YATSM_new" --after class 2000-01-01 LCmap.gtif

  Notes:
      - Image predictions will not use categorical information in timeseries
        models.

Options:
  --root <directory>        Root timeseries directory  [default: ./]
  -r, --result <directory>  Directory of results  [default: YATSM]
  -i, --image <image>       Example timeseries image  [default: example_img]
  --date <format>           Input date format  [default: %Y-%m-%d]
  --ndv <NoDataValue>       Output NoDataValue  [default: -9999]
  -f, --format <driver>     Output format driver  [default: GTiff]
  --warn-on-empty           Warn user when reading in empty results files
  -b, --band <band>         Bands to export for coefficient/prediction maps
  -c, --coef <coef>         Coefficients to export for coefficient maps
  --after                   Use time segment after <date> if needed for map
  --before                  Use time segment before <date> if needed for map
  --qa                      Add QA band identifying segment type
  --refit_prefix TEXT       Use coef/rmse with refit prefix for
                            coefficient/prediction maps  [default: ]
  --amplitude               Export amplitude of sin/cosine pairs instead of
                            individual coefficient estimates
  --predict-proba           Include prediction probability band (scaled by
                            10,000)
  --help                    Show this message and exit.

$ yatsm changemap --help
Usage: yatsm changemap [OPTIONS] <map_type> <start_date> <end_date> <output>

  Examples: TODO

Options:
  --root <directory>        Root timeseries directory  [default: ./]
  -r, --result <directory>  Directory of results  [default: YATSM]
  -i, --image <image>       Example timeseries image  [default: example_img]
  --date <format>           Input date format  [default: %Y-%m-%d]
  --ndv <NoDataValue>       Output NoDataValue  [default: -9999]
  -f, --format <driver>     Output format driver  [default: GTiff]
  --out_date <format>       Output date format  [default: %Y%j]
  --warn-on-empty           Warn user when reading in empty results files
  --magnitude               Add magnitude of change as extra image (pattern is
                            [name]_mag[ext])
  --help                    Show this message and exit.

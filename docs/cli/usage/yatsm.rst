$ yatsm --help
Usage: yatsm [OPTIONS] COMMAND [ARGS]...

  YATSM command line interface

Options:
  --version                Show the version and exit.
  --num_threads <threads>  Number of threads for OPENBLAS/MKL/OMP used in
                           NumPy  [default: 1]
  -v, --verbose            Be verbose
  --verbose-yatsm          Show verbose debugging messages in YATSM algorithm
  -q, --quiet              Be quiet
  -h, --help               Show this message and exit.

Commands:
  cache      Create or update cached timeseries data for YATSM
  changemap  Map change found by YATSM algorithm over time period
  classify   Classify entire images using trained algorithm
  line       Run YATSM on an entire image line by line
  map        Make map of YATSM output for a given date
  pixel      Run YATSM algorithm on individual pixels
  segment    † Warning: could not load plugin. See `build_cli_docs.py segment
             --help`.
  train      Train classifier on YATSM output

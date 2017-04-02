# Example configuration files

This directory contains an example configuration files.

Some of these files use use environment variables to define the paths to
various files. One benefit of using environmental variables is that
you can keep using the same configuration file when running computations
across subsets of data (e.g., MODIS tiles), only changing the definition
of an environment variable to point to the various subsets you want to
process (e.g., `ROOT` could change from `h02v09` to `h02v10`).

## Example:

Let's say we want to define some variable, `ROOT`, that points to
the root directory location where we store some data. We can add
this to our environment when using Bash by exporting the variable:


``` bash
export ROOT=$HOME/Documents/landsat_stack/HFR1_Tower
```

Inside the configuration file, we will reference this environmental
variable with a dollar sign (`$`).


``` yaml

data:
    primary: Landsat
    datasets:
        Landsat:
            reader:
                name: GDAL
                config:
                    input_file: "$ROOT/HFR1_Tower.csv"
```

Here, `input_file: "$ROOT/HFR1_Tower.csv"` will actually qualify to the
value of `$HOME/Documents/landsat_stack/HFR1_Tower/HFR1_Tower.csv`.

Now when you run the script, YATSM will parse this configuration information
and look for images to read from relative to the value of `$ROOT`

``` bash
> yatsm -v batch generic_params.yaml 1 10 
18:27:08 DEBUG yatsm.io.backends._gdal _gdal.read:258 Reading from: /home/ceholden/Documents/landsat_stack/HFR1_Tower/p012r031/LE70120312001161EDC00/LE70120312001161EDC00_stack.tif
```

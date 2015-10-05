# envvar

This directory contains an example configuration file that uses environment
variables to define the paths to various files. This configuration is a copy
of the `p013r030` example file but is included as an example for using
environment variables in configuration files.

## Example:

``` bash
export CONFIG=$HOME/Documents/yatsm/examples/envvar
export ROOTDIR=$HOME/Documents/landsat_stack/p013r030/images
export PICKLES=$HOME/Documents/yatsm/yatsm/regression/pickles
yatsm -v pixel $CONFIG/envvar.yaml 25 25
```

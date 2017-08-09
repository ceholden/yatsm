# envvar

This directory contains an example configuration file that uses environment
variables to define the paths to various files. This configuration is a copy
of the `p013r030` example file but is included as an example for using
environment variables in configuration files.

## Example:

``` bash
# Config location inside "yatsm" respository
export CONFIG=$HOME/Documents/yatsm/examples/envvar
# Cloned and unzipped "landsat_stack" example data
export ROOT=$HOME/Documents/yatsm/landsat_stack/p013r030/images
# Look at a pixel
yatsm -v pixel $CONFIG/envvar.yaml 25 25
```

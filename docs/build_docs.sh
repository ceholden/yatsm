#!/bin/bash

cd $(dirname $0)

# rm yatsm/*.rst
make clean

cd ../

sphinx-apidoc -f -e -o docs/yatsm yatsm/
# sphinx-apidoc -f -e -o docs/scripts scripts/

cd docs/

make html

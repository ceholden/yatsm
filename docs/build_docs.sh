#!/bin/bash

cd $(dirname $0)

cd ../

sphinx-apidoc -f -e -o docs/yatsm yatsm/
# sphinx-apidoc -f -e -o docs/scripts scripts/

cd docs/

make html

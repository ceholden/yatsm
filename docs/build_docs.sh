#!/bin/bash

cd $(dirname $0)

rm yatsm/*.rst
make clean

cd ../

sphinx-apidoc -f -e -o docs/yatsm yatsm/
# sphinx-apidoc -f -e -o docs/scripts scripts/

cd docs/

make html

# copy benchmarks
cd $(dirname $0)/../bench/
asv publish
cp -R html ../docs/_build/html/bench

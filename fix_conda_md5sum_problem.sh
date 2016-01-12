#!/bin/bash
# Fix for md5sum mismatch in Anaconda channel "r"
# See https://github.com/ContinuumIO/anaconda-issues/issues/616

OUT=~/miniconda/pkgs/

wget https://anaconda.org/r/ncurses/5.9/download/linux-64/ncurses-5.9-4.tar.bz2 -O $OUT/ncurses-5.9-4.tar.bz2
wget https://conda.anaconda.org/r/linux-64/glib-2.43.0-2.tar.bz2 -O $OUT/glib-2.43.0-2.tar.bz2
wget https://anaconda.org/r/nlopt/2.4.2/download/linux-64/nlopt-2.4.2-1.tar.bz2 -O $OUT/nlopt-2.4.2-1.tar.bz2
wget https://anaconda.org/r/glib/2.43.0/download/linux-64/glib-2.43.0-2.tar.bz2 -O $OUT/glib-2.43.0-2.tar.bz2

wget https://conda.anaconda.org/r/linux-64/singledispatch-3.4.0.3-py27_1.tar.bz2 -O $OUT/singledispatch-3.4.0.3-py27_1.tar.bz2

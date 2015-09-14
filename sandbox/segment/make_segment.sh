#!/bin/bash
###
#
# Relies on scripts and programs found in following repositories:
#
#   https://github.com/ceholden/misc/
#   https://github.com/ceholden/segment
#
###
set -e

cd p035r032/
rm *

yatsm_map.py -v \
    --root ~/Documents/landsat_stack/p035r032/images/ \
    --after --before \
    predict 2010-06-01 p035r032_2010-06-01_all_predict.gtif

~/Documents/misc/spectral/transforms.py -v -f GTiff \
    p035r032_2010-06-01_all_predict.gtif p035r032_2010-06-01_bgw_predict.gtif \
    brightness greenness wetness

~/Documents/misc/spectral/stretches.py -v \
    --ndv -9999 -ot uint8 \
    p035r032_2010-06-01_bgw_predict.gtif \
    p035r032_2010-06-01_bgw_uint8_predict.gtif \
    percent

# BUG IN SEGMENT RIGHT NOW FOR NON-BIPs
# gdal_translate -of ENVI -co "INTERLEAVE=BIP" \
#     p035r032_2010-06-01_bgw_uint8_predict.gtif temp_bip

# TRY A SMALL SEGMENT TOLERANCE -- skip first phase and then go to the region size restriction auxiliary merging
# If you stretch the all of the data, you're essentially treating all bands the same
#

~/Documents/segment/segment/bin/segment \
    -t 50 -m 0.1 -n 11,11,25,50,250 -8 -o bgw_seg \
    p035r032_2010-06-01_bgw_uint8_predict.gtif

armap=$(ls bgw_seg.armap.*[0-9])
gdal_polygonize.py -8 $armap -f "ESRI Shapefile" armap.shp
rmap=$(ls bgw_seg.rmap.*[0-9])
gdal_polygonize.py -8 $rmap -f "ESRI Shapefile" rmap.shp

rm temp*

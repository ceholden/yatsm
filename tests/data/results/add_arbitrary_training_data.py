#!/usr/bin/env python
""" Create an arbitrary training data image
"""
import os

import numpy as np
import rasterio


here = os.path.dirname(__file__)

example = os.path.join(here, 'example_image.gtif')
training = os.path.join(here, 'training_image_1995-06-01.gtif')


if __name__ == '__main__':
    with rasterio.drivers():
        with rasterio.open(example, 'r') as src:
            meta = src.meta
            meta['nodata'] = 0
            meta['dtype'] = 'uint8'

            # Throw in some labels
            training_img = np.zeros((src.width, src.height), dtype=np.uint8)
            training_img[0, 0:3] = 1
            training_img[1, 0:3] = 2
            training_img[2, 0:3] = 3
            training_img[3, 0:4] = 4
            training_img[4, 0:4] = 5

            with rasterio.open(training, 'w', **meta) as dst:
                dst.write_band(1, training_img)

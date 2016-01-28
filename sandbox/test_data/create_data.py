#!/usr/bin/env python
""" Create test dataset for use in YATSM test suite

Test data derived from the `p035r032` example dataset. See issue #78 for more
information (https://github.com/ceholden/yatsm/issues/78).
"""
from __future__ import division

import errno
import os
import tarfile

import numpy as np
import rasterio

np.random.seed(123456789)

# Output image metadata
dst_crs = {'init': 'EPSG:32613'}
dst_shape = (5, 5)
dst_bands = 8
dst_dtype = np.int16
dst_transform = [336375.0, 30.0, 0.0, 4462425.0, 0.0, -30.0]
dst_nodata = -9999.0
dst_tags = {
    'Band_1': 'Blue x 10,000',
    'Band_2': 'Green x 10,000',
    'Band_3': 'Red x 10,000',
    'Band_4': 'NIR x 10,000',
    'Band_5': 'SWIR1 x 10,000',
    'Band_6': 'SWIR2 x 10,000',
    'Band_7': 'T_b x 100',
    'Band_8': 'Fmask'
}


def mkdir_p(d):
    try:
        os.makedirs(d)
    except OSError as err:
        if err.errno == errno.EEXIST and os.path.isdir(d):
            pass
        else:
            raise


def create_image(root, ID, Y):
    """ Write out an in ``root`` directory image organized by Landsat ID

    Args:
        root (str): root directory
        ID (str): Landsat ID
        Y (np.ndarray): 3D array (nbands x nrows x ncols)
    """
    dst_dir = os.path.join(root, ID)
    mkdir_p(dst_dir)

    dst_img = os.path.join(dst_dir, '%s_stack.gtif' % ID)

    with rasterio.open(dst_img, 'w', driver='GTiff',
                       width=dst_shape[0], height=dst_shape[1],
                       count=dst_bands, dtype=dst_dtype,
                       nodata=dst_nodata,
                       crs=dst_crs, transform=dst_transform) as dst:
        dst.update_tags(**dst_tags)
        dst.write(Y)


def mask_pct(Y, pct, px, py):
    # Add in some % masked pixel
    idx_unmasked = np.where(np.in1d(Y[-1, :, py, px], [0, 1]))[0]
    n_mask = int(n_obs * pct) - idx_unmasked.size
    idx_to_mask = np.random.choice(idx_unmasked, size=n_mask)
    Y[-1, idx_to_mask, py, px] = np.random.choice([2, 3, 4, 255],
                                                  size=idx_to_mask.size)
    return Y


if __name__ == '__main__':
    # Load cache file
    dn = os.path.dirname
    cache_file = os.path.join(dn(dn(dn(os.path.abspath(__file__)))),
                              'tests', 'data', 'cache',
                              'yatsm_r0_n447_b8.npy.npz')
    dat = np.load(cache_file)
    Y, image_IDs = dat['Y'], dat['image_IDs']

    # Setup output
    out_dir = os.path.join(dn(__file__), 'output')
    mkdir_p(out_dir)

    # Reshape and perturb Y data
    n_obs = Y.shape[1]
    n = dst_shape[0] * dst_shape[1]
    Y = Y[..., :n].reshape(Y.shape[0], Y.shape[1], dst_shape[0], dst_shape[1])

    # Add in 100% NODATA pixel into first row
    Y[..., 0, 0] = dst_nodata
    # Add in 100% masked pixels
    Y = mask_pct(Y, 1.00, 1, 0)
    Y = mask_pct(Y, 1.00, 1, 3)
    # Add in 66% masked pixels
    Y = mask_pct(Y, 0.66, 2, 0)
    Y = mask_pct(Y, 0.66, 2, 3)

    # Add in leap year to some image IDs
    image_IDs[image_IDs == 'LT50350321988295XXX04'] = 'LT50350321988366XXX04'
    image_IDs[image_IDs == 'LE70350321999317EDC00'] = 'LE70350321999366EDC00'
    image_IDs[image_IDs == 'LE70350322012289EDC00'] = 'LE70350322012366EDC00'

    # Save to cache files
    cache_dir = os.path.join(out_dir, 'cache')
    mkdir_p(cache_dir)
    for row in range(Y.shape[2]):
        cache_file = os.path.join(cache_dir, 'yatsm_r%i_n447_b8.npy.npz' % row)
        np.savez_compressed(cache_file, **{
            'Y': Y[:, :, row, :],
            'image_IDs': image_IDs
        })

    # Write out to "images" directory
    image_dir = os.path.join(out_dir, 'images')
    for idx_obs in range(Y.shape[1]):
        create_image(image_dir, image_IDs[idx_obs], Y[:, idx_obs, ...])

    # Compress to tar.gz
    with tarfile.open('p035r032_testdata.tar.gz', 'w:gz') as tar:
        tar.add('output', arcname='p035r032')

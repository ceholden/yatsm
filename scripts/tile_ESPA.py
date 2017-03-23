#!/usr/bin/env python
""" Tile data
"""
import errno
import os
from pathlib import Path
import shutil

import click
import delegator
import rasterio

from yatsm.gis.tilespec import TILESPECS


OLI = [
    '*_sr_band2.tif',
    '*_sr_band3.tif',
    '*_sr_band4.tif',
    '*_sr_band5.tif',
    '*_sr_band6.tif',
    '*_sr_band7.tif',
    '*sr_evi.tif',
    '*sr_nbr.tif',
    '*band10.tif',
    '*cfmask.tif'
]
ETM = [
    '*_sr_band*.tif',
    '*_sr_evi.tif',
    '*_sr_nbr.tif',
    '*band6.tif',
    '*cfmask.tif'
]
TM = [
    '*_sr_band*.tif',
    '*_sr_evi.tif',
    '*_sr_nbr.tif',
    '*band6.tif',
    '*cfmask.tif'
]

PATTERNS = {
    'LC8': OLI,
    'LT4': TM,
    'LT5': TM,
    'LE7': ETM
}


CMD_GDALBUILDVRT = (
    'gdalbuildvrt '
    '-separate '
    '-te {bounds.left} {bounds.bottom} {bounds.right} {bounds.top} '
    '-overwrite '
    '{vrt} '
    '{images}'
)


def find_images(xml, patterns):
    for pattern in patterns:
        results = xml.parent.glob(pattern)
        for result in sorted(results):
            yield result


def relative_to(one, other):
    root = '/'
    for parent in one.parents:
        if parent in other.parents:
            root = parent
            break
    fwd = one.relative_to(root)
    bwd = ('..', ) * (len(other.relative_to(root).parents) - 1)
    return Path('.').joinpath(*bwd).joinpath(fwd)


@click.command(short_help='Tile some data')
@click.argument('srcs', nargs=-1,
                type=click.Path(readable=True, file_okay=True,
                                dir_okay=False, resolve_path=True))
@click.argument('dst',
                type=click.Path(writable=True, file_okay=False,
                                resolve_path=True))
@click.option('--tilespec_name', default='LCMAP_ARD',
              type=click.Choice(TILESPECS.keys()), show_default=True,
              help='Tile specification')
@click.option('--absolute/--relative', default=True, show_default=True,
              help='Write VRTs using absolute paths or paths relative to DST')
@click.option('--suffix', default='_stack.vrt', show_default=True,
              help='VRT filename suffix (after $(basename SRC .xml) is used)')
@click.pass_context
def tile_ESPA(ctx, srcs, dst, tilespec_name, absolute, suffix):
    """ Create VRT tiles from some Landsat ESPA products

    NOTE: very "hard-coded" currently, but WIP to be generic
    """
    tilespec = TILESPECS[tilespec_name]

    dst = Path(dst)
    srcs = [Path(src) for src in srcs]

    n_expected_bands = None

    for src in srcs:
        if src.suffix != '.xml':
            raise click.BadParameter('Must specify ESPA XML metadata files as '
                                     'SRCS...: {1}'.format(str(src)))
        pattern = PATTERNS[src.name[0:3]]
        imgs = list(find_images(src, pattern))
        if not imgs:
            click.echo('Could not find image for "{0}"'.format(str(src)))
            continue

        if n_expected_bands is None:  # set first time
            n_expected_bands = len(imgs)
        else:
            if len(imgs) != n_expected_bands:
                click.echo('Product "{0}" has a different number of bands '
                           'compared to expectation from first result '
                           '({1} vs {2})'
                           .format(str(src), len(imgs), n_expected_bands))
                continue

        # OK good to go
        with rasterio.open(str(imgs[0]), 'r') as example_ds:
            bounds = example_ds.bounds
        tiles = list(tilespec.bounds_to_tiles(bounds))
        for tile in tiles:
            tile_hv = 'h{0:02d}v{1:02d}'.format(tile.horizontal,
                                                tile.vertical)
            dst_dir = dst.joinpath(tile_hv, src.stem)
            try:
                dst_dir.mkdir(parents=True)
            except OSError as oserr:
                if oserr.errno != errno.EEXIST:
                    raise

            vrt = dst_dir.joinpath(src.stem + suffix)

            if not absolute:
                os.chdir(str(vrt.parent))
                imgs = [relative_to(img, vrt) for img in imgs]

            cmd_str = CMD_GDALBUILDVRT.format(
                bounds=tile.bounds,
                vrt=vrt if absolute else vrt.name,
                images=' '.join([str(img) for img in imgs])
            )

            cmd = delegator.run(cmd_str)
            if cmd.return_code:
                click.echo('Error writing to: {0}'.format(vrt))
                click.echo(cmd.err)
            else:
                click.echo('Wrote to: {0}'.format(vrt))

            # Copy ESPA metadata
            shutil.copy(str(src), str(dst_dir.joinpath(src.name)))
            # Copy MTL if any
            mtl = list(src.parent.glob('L*MTL.txt'))
            if mtl:
                mtl = mtl[0]
                shutil.copy(str(mtl), str(dst_dir.joinpath(mtl.name)))

    click.echo('Done')


if __name__ == '__main__':
    tile_ESPA()

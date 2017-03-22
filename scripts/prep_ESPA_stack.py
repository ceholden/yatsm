#!/usr/bin/env python
""" Prepare a "stacked" ESPA dataset for YATSM GDALTimeSeries use
"""
from __future__ import print_function

from collections import OrderedDict
import datetime as dt
import os
import re
try:
    from bs4 import BeautifulSoup
    import click
    from pathlib import Path
    import pandas as pd
except ImportError as ie:
    print('Error! Cannot run script {0} without necessary library:'
          .format(os.path.basename(__file__)))
    raise


DATETIME = 'datetime'


def find(d, s=''):
    for dirpath, dirnames, filenames in os.walk(str(d)):
        for filename in filenames:
            p = Path(os.path.join(dirpath, filename))
            if p.is_file() and (p.match(s) if s else True):
                yield p.resolve()


def extract_md(soup):
    # Old XML files don't give the Landsat ID ;-(
    _id = soup.find('product_id') or soup.find('lpgs_metadata_file')
    _id = _id.text.split('_MTL.txt')[0]

    ad = soup.find('acquisition_date').text
    ct = soup.find('scene_center_time').text
    datetime = '{}T{}'.format(ad, ct)

    sat = soup.find('satellite').text
    sat_num = re.split(r'(_|\s)', sat)[-1]
    instrument = soup.find('instrument').text

    angles = soup.find('solar_angles').attrs
    sza, saa = angles['zenith'], angles['azimuth']

    return OrderedDict((
        ('id', _id),
        (DATETIME, datetime),
        ('satellite', sat),
        ('instrument', instrument),
        ('sensor', '{0}{1}'.format(instrument, sat_num)),
        ('solar_zeniith', sza),
        ('solar_azimuth', saa)
    ))


@click.command()
@click.argument('root', type=click.Path(file_okay=False, exists=True))
@click.argument('output', type=click.Path(dir_okay=False, writable=True),
                required=False)
@click.option('-o', '--to', type=click.Path(dir_okay=False, writable=True))
@click.option('--img_pattern', default='*stack.*tif', show_default=True,
              type=str,
              help='Image pattern to search for adjacent to XML files')
@click.option('--absolute/--relative', default=True,
              help='Use absolute or paths')
@click.pass_context
def prep_ESPA_stack(ctx, root, output, to, img_pattern, absolute):
    output = output or to
    if output:  # otherwise goes to stdout
        output = Path(output).resolve()
    elif not output and not absolute:
        click.echo('Cannot determine relative paths to STDOUT')
        raise click.Abort()

    df = []
    for _xml in find(root, 'L*.xml'):
        if str(_xml).endswith('aux.xml'):
            continue
        with open(str(_xml), 'r') as f:
            # TODO: relative paths?
            image = list(_xml.parent.glob(img_pattern))[0]
            soup = BeautifulSoup(f.read(), 'lxml')

            row = OrderedDict()
            row['filename'] = str(image.resolve() if absolute else
                                  image.relative_to(output.parent))
            row.update(extract_md(soup))
            df.append(row)

    df = pd.DataFrame.from_records(df).sort_index()
    df[DATETIME] = pd.to_datetime(df[DATETIME])
    _csv = df.to_csv(output, header=True, index=False, mode='w',
                     date_format='%Y-%m-%dT%H:%M:%S.%f')
    click.echo(_csv)


if __name__ == '__main__':
    prep_ESPA_stack()

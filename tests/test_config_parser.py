""" Tests for yatsm.config_parser
"""
import os

from yatsm import config_parser


def test_get_envvars():
    truth = {
        'YATSM': {
            'algo': 'CCDC',
            'jobno': '1'
        },
        'dataset': {
            'dataset': '/tmp/images.csv',
            'cache': '/tmp/cache'
        }
    }
    d = {
        'YATSM': {
            'algo': 'CCDC',
            'jobno': '$JOBNO'
        },
        'dataset': {
            'dataset': '$ROOTDIR/images.csv',
            'cache': '$ROOTDIR/cache'
        }
    }
    envvars = {
        'JOBNO': '1',
        'ROOTDIR': '/tmp'
    }
    # Backup and replace environment
    backup = os.environ.copy()
    for k in envvars:
        os.environ[k] = envvars[k]

    expanded = config_parser.expand_envvars(d)

    os.environ.update(backup)

    assert truth == expanded

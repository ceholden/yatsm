#!/usr/bin/env python
""" Near real-time monitoring

Usage:
    monitor_yatsm.py [options] <config_file> <job_number> <total_jobs>

Options:
    --resume                    Do not overwrite pre-existing results
    -v --verbose                Show verbose debugging messages
    --version                   Print program version and exit
    -h --help                   Show help

"""
import logging

FORMAT = '%(asctime)s:%(levelname)s:%(module)s.%(funcName)s:%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H-%M-%S')
logger = logging.getLogger('yatsm')

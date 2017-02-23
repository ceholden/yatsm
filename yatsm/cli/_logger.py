""" Nice click based logging
"""
import copy
import logging

import click

# Configure logging
DEFAULT_LOG_FORMAT = ('%(asctime)s %(levelname)s %(name)s '
                       '%(module)s.%(funcName)s:%(lineno)s '
                       '%(message)s')
DEFAULT_LOG_TIME_FORMAT = '%H:%M:%S'


class ColorFormatter(logging.Formatter):
    colors = {
        'debug': dict(fg='blue'),
        'info': dict(fg='green'),
        'warning': dict(fg='yellow', bold=True),
        'error': dict(fg='red', bold=True),
        'exception': dict(fg='red', bold=True),
        'critical': dict(fg='red', bold=True)
    }

    def format(self, record):
        if not record.exc_info:
            record = copy.copy(record)
            style = self.colors.get(record.levelname.lower(), {})
            record.levelname = click.style(record.levelname, **style)
        return logging.Formatter.format(self, record)


class ClickHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            click.echo(msg, err=True)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def config_logging(level, config=None):
    if config:
        logging.config.dictConfig(config)
    else:
        handler = ClickHandler()
        formatter = ColorFormatter(DEFAULT_LOG_FORMAT,
                                   DEFAULT_LOG_TIME_FORMAT)
        handler.setFormatter(formatter)

        logger = logging.getLogger('yatsm')
        logger.addHandler(handler)
        logger.setLevel(level)

        logger_algo = logging.getLogger('yatsm.algo')
        logger_algo.addHandler(handler)
        logger_algo.setLevel(level + 10)

        if level <= logging.INFO:  # silence rasterio
            logging.getLogger('rasterio').setLevel(logging.INFO)


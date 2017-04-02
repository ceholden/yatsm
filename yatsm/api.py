""" Helper class for dealing with configuration files
"""
import copy
import functools
import itertools
import logging

from yatsm.config import load_config, parse_config, validate_config
from yatsm.errors import (InvalidConfigurationException,
                          PipelineConfigurationNotFound)
from yatsm.io import get_readers
from yatsm.pipeline import Pipe, Pipeline
from yatsm.results import HDF5ResultsStore
from yatsm.results.utils import pattern_to_regex as _pattern_to_regex
from yatsm.utils import cached_property as _cached_property, find as _find

logger = logging.getLogger(__name__)


def _checked_property(prop):
    """ Check a :class:`yatsm.Config` for a certain property
    """
    @functools.wraps(prop)
    def wrapper(self):
        if prop.__name__ not in self.keys():
            raise PipelineConfigurationNotFound('Section "{0}" not found'
                                                .format(prop.__name__))
        return prop(self)
    return property(wrapper)


class Config(object):  # TODO: rename?
    """ Your go-to list of things to do

    Args:
        config (dict): A `dict` representing a YATSM configuration file
        validate (bool): Valdiate :ref:`config` against expected schema
    """

    SECTIONS = ('version', 'pipeline', 'tasks', 'results', 'data', )

    def __init__(self, config, validate=True):
        if validate:
            config = validate_config(config)
        self._config = copy.deepcopy(config)

    @classmethod
    def from_file(cls, filename):
        """ Return this configuration as customized ``dict`` container
        """
        data = load_config(filename)
        config = validate_config(parse_config(data))
        return cls(config)

    # READERS
    @_cached_property
    def readers(self):
        return get_readers(self._config['data']['datasets'])

    @property
    def primary_reader(self):
        pref = (self.data.get('primary', '') or
                list(self.readers.keys())[0])
        return self.readers[pref]

    # PIPELINE
    def get_pipeline(self, pipe=None, overwrite=False):
        """ Return a :ref:`yatsm.pipeline.Pipeline`

        Args:
            pipe (yatsm.pipeline.Pipe): Pipe data object
            overwrite (bool): Allow overwriting

        Returns:
            yatsm.pipeline.Pipeline: YATSM pipeline
        """
        pipe = pipe or Pipe()
        return Pipeline.from_config(self.tasks, pipe, overwrite=overwrite)

    # RESULTS
    def find_results(self, output=None, output_prefix=None, **kwds):
        """ A list of :ref:`HDF5ResultsStore` results
        """
        output = output or self.results['output']
        output_prefix = output_prefix or self.results['output_prefix']

        pattern = _pattern_to_regex(output_prefix or
                                    self.results.get('output_prefix'))
        results = _find(output, pattern, regex=True)
        for result in results:
            yield HDF5ResultsStore(result, **kwds)

    def peak_results(self, results, table=None):
        """ Pop an example result file off a list to inspect

        Args:
            results (iterable): A generator or list of
                :class:`HDF5ResultsStore`
            table (str): Optionally, table to require in order to be a "valid"
                example

        Returns:
            tuple: Return an example :class:`HDF5ResultsStore` and the original
            generator (or list) supplied to the function (with the example
            still in the generator)

        Raises:
            StopIteration: Raise (by not catching if thrown) if function cannot
            find any valid results
        """
        result = None
        _results = []  # storage for iterator's items we've seen
        while result is None:
            candidate = next(results)
            _results.append(candidate)  # put back

            try:
                tables = list(candidate.tables())
                if not tables:
                    logger.debug('No results to peak in {0}'
                                 .format(candidate.filename))
                elif table and table not in tables:
                    logger.debug('Table "{0}" not found in file {1}'
                                 .format(table, candidate.filename))
                else:
                    result = candidate
            except Exception as exc:
                logger.debug('Could not use "{0}" as example result file: '
                             '{1!r}'.format(candidate.filename, exc))

        return result, itertools.chain(_results, results)

    def peak_table(self, store):
        """ Return the first obvious table to use from a HDF5ResultsStore

        Args:
            store (HDF5ResultsStore): HDF5 results file

        Returns:
            str or None: table location, or `None` if there are no tables
            stored
        """
        tables = list(store.tables())
        if tables:
            table_name = tables[0][0]
            return table_name

    # PROPERTY ACCESS
    @_checked_property
    def pipeline(self):
        """ Pipeline configuration section
        """
        return self._config['pipeline']

    @property
    def tasks(self):
        return self.pipeline['tasks']

    @_checked_property
    def data(self):
        return self._config['data']

    @_checked_property
    def version(self):
        return self._config['version']

    @_checked_property
    def results(self):
        return self._config['results']

    # DICT-LIKE ACCESS
    def keys(self):
        return self._config.keys()

    def __getitem__(self, key):
        if key not in self.SECTIONS:
            raise KeyError('Unknown configuration item "{0}"'.format(key))
        try:
            return self._config[key]
        except KeyError as ke:
            raise PipelineConfigurationNotFound('Section "{0}" not found'
                                                .format(key))

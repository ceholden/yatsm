""" Helper class for dealing with configuration files
"""
import copy

from yatsm.config import validate_and_parse_configfile
from yatsm.utils import cached_property as _cached_property, find as _find
from yatsm.io import get_readers
from yatsm.results import HDF5ResultsStore
from yatsm.results.utils import pattern_to_regex as _pattern_to_regex
from yatsm.pipeline import Pipe, Pipeline


class Config(object):  # TODO: rename?
    """ Your go-to list of things to do
    """

    def __init__(self, config):
        self._config = copy.deepcopy(config)

    @classmethod
    def from_file(cls, filename):
        """ Return this configuration as customized ``dict`` container
        """
        c = validate_and_parse_configfile(filename)
        return cls(c)

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

    # PROPERTY ACCESS
    @property
    def data(self):
        return self._config['data']

    @property
    def pipeline(self):
        return self._config['pipeline']

    @property
    def tasks(self):
        return self._config['pipeline']['tasks']

    @property
    def version(self):
        return self._config['version']

    @property
    def results(self):
        return self._config['results']

    # DICT-LIKE ACCESS
    def keys(self):
        return self._config.keys()

    def __getitem__(self, key):
        return self._config[key]

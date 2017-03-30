from yatsm.version import (__docs_version__,
                           __repo_issues__,
                           __version__)


class TSLengthException(Exception):
    """ Exception stating timeseries does not contain enough observations
    """
    pass


class TrainingDataException(Exception):
    """ Custom exception for errors with training data """
    pass


class AlgorithmNotFoundException(Exception):
    """ Custom exception for algorithm config files without handlers """
    pass


class InvalidConfigurationException(Exception):
    pass


class PipelineConfigurationError(TypeError):
    """ Exception for invalid ``require``/``output`` specification in pipeline
    """
    pass


class PipelineConfigurationNotFound(KeyError):
    """ Exception for uses requiring missing information

    Provides help message with link to configuration docs.
    """
    # TODO: finish help message & docs
    _help = ('Missing configuration information required to accomplish task. '
             'Please consult {0}/{1} for more information'
             .format(__docs_version__, 'guide/pipeline.html#errors'))

    def __init__(self, msg, *args, **kwds):
        msg = '{0} : "{1!r}"'.format(self._help, msg)
        super(KeyError, self).__init__(msg, *args[1:], **kwds)


class TODO(NotImplementedError):

    _help = ('This version ("{version}") has not yet implemented this '
             'functionality. Please visit {issues} and copy and paste '
             'the following message if you would like this feature to be '
             'supported.'
             .format(version=__version__, issues=__repo_issues__))

    def __init__(self, msg, *args, **kwds):
        msg = '{0} : "{1}"'.format(self._help, msg)
        super(NotImplementedError, self).__init__(msg, *args[1:], **kwds)

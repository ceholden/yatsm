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

import logging

_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)

logger = logging.getLogger('yatsm')
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

logger_algo = logging.getLogger('yatsm_algo')
logger_algo.addHandler(_handler)
logger_algo.setLevel(logging.WARNING)

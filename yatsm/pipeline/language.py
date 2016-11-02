""" Storage for the language used in this pipeline batch processing
"""
#: str: Name of argument that species task's Python function
TASK = 'task'

#: str: Name of function argument for specifying task requirements
REQUIRE = 'require'

#: str: Name of function argument for specifying task outputs
OUTPUT = 'output'

#: str: Name of variable containing results computed during the pipeline
PIPE = 'pipe'

#: str: Name of variable containing task-specific configuration
CONFIG = 'config'

#: str: Name of raster "data" naming convention
DATA = 'data'

#: str: Name of segment "structure" or "record" database-like information
RECORD = 'record'

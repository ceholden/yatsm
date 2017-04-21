""" Task attributes set as function attributes
"""


def segment_task(func):
    """ A task decorator that declares it creates segments
    """
    func.is_segmenter = True
    return func


def eager_task(func):
    """ A task decorator that declares it can compute all pixels at once
    """
    func.is_eager = True
    return func


def task_version(version_str):
    def decorator(func):
        func.version = version_str
        return func
    return decorator

""" Raster data reading utilities
"""


def block_windows(block_shape, shape):
    """ Returns an iterator over block window ids and windows

    Args:
        block_shape (tuple): Shape of blocks
        shape (tuple): Shape of the image

    Yields:
        tuple: ``(block, window)``
    """
    h, w = block_shape
    height, width = shape

    d, m = divmod(height, h)
    nrows = d + int(m > 0)
    d, m = divmod(width, w)
    ncols = d + int(m > 0)

    for j in range(nrows):
        row = j * h
        height = min(h, height - row)
        for i in range(ncols):
            col = i * w
            width = min(w, width - col)
            yield (j, i), ((row, row+height), (col, col+width))

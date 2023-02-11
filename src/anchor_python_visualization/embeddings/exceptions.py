"""Custom exception-types for this package."""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"


class InsufficientRowsException(Exception):
    """When there are too few rows in a data-frame to perform an operation."""

    pass

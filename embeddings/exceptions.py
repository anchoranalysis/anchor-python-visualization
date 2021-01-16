"""Custom exception-types for this package."""


class InsufficientRowsException(Exception):
    """Raised when there are too few rows in the data-frame for a particular operation."""
    pass

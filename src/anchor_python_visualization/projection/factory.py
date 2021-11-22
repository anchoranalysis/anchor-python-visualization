"""Creates instances of :class:`~projection.Projector`."""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

from typing import Optional

from ._pca import PCAProjection
from ._tsne import TSNEProjection
from .projector import Projector


IDENTIFIERS = ["t-SNE", "PCA", "none"]
"""Unique strings to use as command-line-arguments to select a :class:`Projector`."""


DEFAULT_IDENTIFIER = "t-SNE"
"""The default choice to use in :const:`IDENTIFIERS`."""


def create_projector(identifier: str) -> Optional[Projector]:
    """
    Creates a projection method from an identifier.

    :param identifier: string that is one of :const:`IDENTIFIERS`
    :returns: a newly created projection method, or none at all.
    """
    if identifier == IDENTIFIERS[0]:
        return TSNEProjection()
    elif identifier == IDENTIFIERS[1]:
        return PCAProjection()
    elif identifier == IDENTIFIERS[2]:
        return None
    else:
        raise ValueError("Unknown identifier for projection: {}".format(identifier))

from typing import Optional

from ._pca import PCAProjection
from ._tsne import TSNEProjection
from .projector import Projector

DEFAULT_IDENTIFIER = "t-SNE"
IDENTIFIERS = ["t-SNE", "PCA", "none"]


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
        raise Exception(
            "Unknown identifier for projection: {}".format(identifier)
        )

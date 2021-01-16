from typing import Optional

from ._pca import PCAProjection
from ._tsne import TSNEProjection
from .projection import Projection

DEFAULT_IDENTIFIER = "t-SNE"
IDENTIFIERS = ["t-SNE", "PCA", "none"]


def create_projection_method(method_identifier: str) -> Optional[Projection]:
    """
    Creates a projection method from an identifier.

    :param method_identifier: string that is one of :const:`IDENTIFIERS`
    :returns: a newly created projection method, or none at all.
    """
    if method_identifier == IDENTIFIERS[0]:
        return TSNEProjection()
    elif method_identifier == IDENTIFIERS[1]:
        return PCAProjection()
    elif method_identifier == IDENTIFIERS[2]:
        return None
    else:
        raise Exception(
            "Unknown identifier for projection: {}".format(method_identifier)
        )

from typing import Optional

from ._pca import PCAProjection
from ._tsne import TSNEProjection
from .projection import Projection

PROJECTION_FACTORY_DEFAULT_IDENTIFIER = "t-SNE"
PROJECTION_FACTORY_IDENTIFIERS = [PROJECTION_FACTORY_DEFAULT_IDENTIFIER, "PCA", "none"]


def create_projection_method(method_identifier: str) -> Optional[Projection]:
    """
    Creates a projection method from an identifier
    :param method_identifier: string that is one of PROJECTION_FACTORY_IDENTIFIERS
    :return: a newly created projection method, or none at all
    """
    if method_identifier == "t-SNE":
        return TSNEProjection()
    elif method_identifier == "PCA":
        return PCAProjection()
    elif method_identifier == "none":
        return None
    else:
        raise Exception(
            "Unknown identifier for projection: {}".format(method_identifier)
        )

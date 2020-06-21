from typing import Optional
from ._tsne import TSNEProjection
from ._pca import PCAProjection


PROJECTION_FACTORY_DEFAULT_IDENTIFIER = "t-SNE"
PROJECTION_FACTORY_IDENTIFIERS = [PROJECTION_FACTORY_DEFAULT_IDENTIFIER, "PCA"]


def create_projection_method(method_identifier: Optional[str]):
    """
    Creates a projection method from an identifier
    :param method_identifier: string that is one of PROJECTION_FACTORY_IDENTIFIERS
    :return: a newly created projection method
    """
    if method_identifier == "t-SNE" or id is None:
        return TSNEProjection()
    elif method_identifier == "PCA":
        return PCAProjection()
    else:
        raise Exception(
            "Unknown identifier for projection: {}".format(method_identifier)
        )
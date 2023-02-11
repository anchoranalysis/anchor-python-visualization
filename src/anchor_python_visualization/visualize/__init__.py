"""Different schemes for visualizing embeddings or other types of data."""
from .factory import DEFAULT_IDENTIFIER, IDENTIFIERS, create_method
from .visualize_features_scheme import VisualizeFeaturesScheme

__all__ = [
    "DEFAULT_IDENTIFIER",
    "IDENTIFIERS",
    "create_method",
    "VisualizeFeaturesScheme",
]

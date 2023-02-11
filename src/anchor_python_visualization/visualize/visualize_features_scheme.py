"""Abstract base class for a scheme for visualizing embeddings"""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

from abc import ABC

from anchor_python_visualization import embeddings


class VisualizeFeaturesScheme(ABC):
    """A method to visualize a set of feature-values."""

    def visualize_data_frame(self, features: embeddings.LabelledFeatures) -> None:
        """Visualizes the embeddings in some manner.

        Args:
            features: embeddings in a data-frame with associated labels.
        """
        pass

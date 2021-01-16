"""Abstract base class for a scheme for visualizing embeddings"""
from abc import ABC

import embeddings


class VisualizeFeaturesScheme(ABC):
    """An method to visualize a set of feature-values."""

    def visualize_data_frame(self, features: embeddings.LabelledFeatures) -> None:
        """
        Performs some form of visualization of embeddings.

        :param features: embeddings in a data-frame with associated labels.
        """
        pass

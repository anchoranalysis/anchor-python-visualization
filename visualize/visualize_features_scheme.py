"""Abstract base class for a scheme for visualizing features"""
from abc import ABC

from features import LabelledFeatures


class VisualizeFeaturesScheme(ABC):
    """An method to visualize a set of feature-values."""

    def visualize_data_frame(self, features: LabelledFeatures) -> None:
        """
        Performs some form of visualization of features.

        :param features: features in a data-frame with associated labels.
        """
        pass

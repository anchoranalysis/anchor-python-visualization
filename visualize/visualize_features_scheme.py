"""Abstract base class for a scheme for visualizing features"""
from abc import ABC

from features import LabelledFeatures


class VisualizeFeaturesScheme(ABC):
    """An approach to visualizing a table of feature-values"""

    def visualize_data_frame(self, features: LabelledFeatures) -> None:
        """Performs some form of visualization on a data-frame

        Arguments:
        ----------
        features:
            features in a data-frame and associated labels
        """
        pass

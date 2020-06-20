from abc import ABC
from _labelled_features import LabelledFeatures


class VisualizeScheme(ABC):
    """One approach to visualizing a table of feature-values"""

    def visualize_data_frame(self, features: LabelledFeatures) -> None:
        """Performs some form of visualization on a data-frame

        Arguments:
        ----------
        features:
            features in a data-frame and associated labels
        """
        pass

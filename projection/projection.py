"""Abstract base class for projection method"""
from abc import ABC

import pandas as pd


class Projection(ABC):
    """Projects the feature-space to lower dimensionality."""

    def project(self, features: pd.DataFrame) -> pd.DataFrame:
        """Performs projection, while preserving a data-frame with identical row names.

        :param features: data_frame containing only numerical features (as columns) and with labelled row.names
        :return: a data-frame of features with identical order and row names, but changed columns.
        """
        pass

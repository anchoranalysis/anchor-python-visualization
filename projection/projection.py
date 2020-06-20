"""Abstract base class for projection method"""
from abc import ABC

import pandas as pd


class Projection(ABC):
    """Projects the feature-space to lower dimensionality"""

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs projection, while preserving a data-frame with identical row names

        Arguments:
        ----------
        df:
            data_frame containing only numerical features (as columns) and with labelled row.names

        Returns:
        --------
        a data_frame with identical order and row.names
        """
        pass

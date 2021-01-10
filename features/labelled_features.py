"""Routines for loading features from CSV and adding identifiers and labels"""
import random
from typing import Optional
from .exceptions import InsufficientRowsException

import pandas as pd


class LabelledFeatures:
    """Maintains separate data-frames for features and labels, but with the same number and order of rows"""

    def __init__(self, features: pd.DataFrame, labels: pd.Series, image_paths: Optional[pd.Series] = None):
        """Constructor

        :param features: data-frame containing only feature-values (all numeric), and with each row assigned an identifier
        :param labels: series with labels for each item in df_features (the series must have the same size and order as ``df_features``)
        :param image_paths: optional series with a path to an image for each item (the series must have the same size and order as ``df_features``)
        """
        self.features = features
        self.labels = labels
        self.image_paths = image_paths

    def number_items(self) -> int:
        """Returns the number of items (i.e. rows) in the data-frames/series"""
        return len(self.features.index)

    def sample_without_replacement(self, sample_size: int) -> 'LabelledFeatures':
        """Samples without replacement (taking identical rows from each member data-frame/series)

        @param sample_size: number of items to sample
        @param replace: allow or disallow sampling of the same row more than once.
        @:raise InsufficientRowsException if there are fewer rows available than n
        """
        number_rows = self.number_items()
        if sample_size > number_rows:
            raise InsufficientRowsException("Cannot sample {} rows from a data-frame with only {} rows", sample_size,
                                            number_rows)
        elif sample_size == number_rows:
            # Nothing to do
            return
        else:
            indices = random.sample(range(number_rows), sample_size)
            return LabelledFeatures(
                self.features.iloc[indices, :],
                self.labels.iloc[indices],
                self.image_paths.iloc[indices]
            )
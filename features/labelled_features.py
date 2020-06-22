"""Routines for loading features from CSV and adding identifiers and labels"""
from typing import Optional

import pandas as pd
import random


class LabelledFeatures:
    """Maintains separate data-frames for features and labels, but with the same number and order of rows"""

    def __init__(self, df_features: pd.DataFrame, labels: pd.Series, image_paths: Optional[pd.Series] = None):
        """Constructor

        :param df_features: data-frame containing only feature-values (all numeric), and with each row assigned an identifier
        :param labels: series with labels for each item in df_features (the series must have the same size and order as df_features)
        :param image_paths: optional series with a path to an image for each item (the series must have the same size and order as df_features)
        """
        self.df_features = df_features
        self.labels = labels
        self.image_paths = image_paths

    def num_items(self) -> int:
        """Returns the number of items (i.e. rows) in the data-frames/series"""
        return len(self.df_features.index)

    def sample_without_replacement(self, n: int) -> 'LabelledFeatures':
        """Samples without replacement (taking identical rows from each member data-frame/series)

        @param n: number of items to sample
        @param replace: allow or disallow sampling of the same row more than once.
        @:raise Exception if threre are fewer rows available than n
        """
        num_rows = self.num_items()
        if n > num_rows:
            raise Exception("Cannot sample {} rows from a data-frame with only {} rows", n, num_rows)
        elif n==num_rows:
            # Nothing to do
            return
        else:
            indices = random.sample(range(num_rows), n)
            return LabelledFeatures(
                self.df_features.iloc[indices,:],
                self.labels.iloc[indices],
                self.image_paths.iloc[indices]
            )
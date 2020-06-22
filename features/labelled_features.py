"""Routines for loading features from CSV and adding identifiers and labels"""
from typing import Optional

import pandas as pd


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

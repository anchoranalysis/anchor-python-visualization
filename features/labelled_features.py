"""Routines for loading features from CSV and adding identifiers and labels"""
from typing import Optional
import pandas as pd


class LabelledFeatures:
    """Maintains separate data-frames for features and labels, but with the same number and order of rows"""

    def __init__(self, df_features: pd.DataFrame, labels: Optional[pd.Series]):
        """Constructor

        :param df_features: data-frame containing only feature-values (all numeric), and with each row assigned an identifier
        :param labels: optional series with labels for each item in df_features (the series must have the same size and order)
        """
        self.df_features = df_features
        self.labels = labels

"""Routines for loading features from CSV and adding identifiers and labels"""
import argparse
from typing import Optional

import numpy as np
import pandas as pd

# Name for index column
COL_NAME_INDEX = 'identifier'


class LabelledFeatures:
    """Maintains separate data-frames for features and labels, but with the same number and order of rows"""

    def __init__(self, df_features: pd.DataFrame, labels: Optional[pd.Series]):
        """Constructor

        Arguments:
        ---------
        df_features:
            data-frame containing only feature-values (all numeric), and with each row assigned an identifier
        labels:
            optional series with labels for each item in df_features (the series must have the same size and order)
        """
        self.df_features = df_features
        self.labels = labels


def load_features(args: argparse.ArgumentParser) -> LabelledFeatures:
    """Loads the features from a CSV file, determines identifiers and labels - all according to the arguments"""

    # Read all columns, text and number
    df = pd.read_csv(args.file_path_to_csv, index_col=None, header=0)

    # Find the numeric and string columns
    df_numeric_cols, df_string_cols = df.select_dtypes(include=np.number), df.select_dtypes(include=['object'])

    # Extract or create identifiers for the data-frame
    identifiers = _select_or_create_identifiers(df_string_cols)

    df_with_identifiers = _add_row_names(df_numeric_cols, identifiers)

    # Take the first string col as the row names (index)
    return LabelledFeatures(
        df_with_identifiers,
        _derive_first_group_label_from_identifiers(df_with_identifiers)
    )


def _select_or_create_identifiers(df_string_cols) -> pd.Series:
    """Selects the first (left-most) string column as the identifiers or otherwise creates a range of numbers"""
    if len(df_string_cols.columns) > 0:
        return df_string_cols.iloc[:, 0]
    else:
        return pd.Series(
            range(
                len(df_string_cols.columns)
            )
        )


def _add_row_names(df: pd.DataFrame, row_names: pd.Series) -> pd.DataFrame:
    """Adds a series as row-names to a data-frame"""
    df[COL_NAME_INDEX] = row_names
    df.set_index(COL_NAME_INDEX, inplace=True)
    return df


def _derive_first_group_label_from_identifiers(df: pd.DataFrame) -> pd.Series:
    """Derives the first group (leftmost group in name) from the names of a data-frame"""
    row_names = df.index.values

    def extract_first_group(name):
        return name.split("/")[0]

    return pd.Series(
        list(map(extract_first_group, row_names)),
        dtype="category",
        index=df.index
    )
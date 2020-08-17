"""Loading and labelling features"""
import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from features import LabelledFeatures

# Name for index column
COL_NAME_INDEX = 'identifier'

# Optional placeholder used in image_dir argument
PLACEHOLDER_FOR_SUBSTITUTION = '<IMAGE>'


def load_features(args: argparse.ArgumentParser) -> LabelledFeatures:
    """Loads the features from a CSV file, determines identifiers and labels - all according to the arguments"""

    # Read all columns, text and number
    df = pd.read_csv(args.file_path_to_csv, index_col=None, header=0, encoding=args.encoding)

    # Find the numeric and string columns
    df_numeric_cols = df.select_dtypes(include=np.number)
    df_string_cols = df.select_dtypes(include=['object'])

    # Extract or create identifiers for the data-frame
    identifiers = _select_or_create_identifiers(df_string_cols)

    df_with_identifiers = _add_row_names(df_numeric_cols.copy(), identifiers)

    # Take the first string col as the row names (index)
    return LabelledFeatures(
        df_with_identifiers,
        _derive_first_group_label_from_identifiers(df_with_identifiers),
        _maybe_image_paths(args.image_dir, df_with_identifiers)
    )


def _maybe_image_paths(image_dir: Optional[str], df: pd.DataFrame) -> pd.Series:
    """
    Maybe creates a series of image-paths derived from the index names in df (the returned series has identical size and order)

    No paths are created if image-dir is None, and instead None is returned.

    @param image_dir iff present, a directory of images in which the index names of the df refer to an image inside the directory.
    """
    if image_dir is None:
        return None

    return df.index.to_series().map(
        lambda path: _join_or_substitute(image_dir, path)
    )


def _join_or_substitute(image_dir: str, path: str) -> str:
    """
    Either joins path to image_dir or substitutes path into image_dir (if it contains PLACEHOLDER_FOR_SUBSTITUTION)

    Both paths are normed so that directory-seperators match the execution environment.

    :param image_dir: either the root directory where images exist OR a full path with a placeholder PLACEHOLDER_FOR_SUBSTITUTION which can be substituted
    :param path: the relative-path to an image
    :return: either the relative-path joined to image_dir or the relative-path substituted into image_dir in place of PLACEHOLDER_FOR_SUBSTITUTION
    """
    if PLACEHOLDER_FOR_SUBSTITUTION in image_dir:
        return os.path.normpath(image_dir).replace(
            PLACEHOLDER_FOR_SUBSTITUTION,
            os.path.normpath(path),
            1
        )
    else:
        return os.path.join(image_dir, path)


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

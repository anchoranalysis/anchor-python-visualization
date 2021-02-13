"""Loading and labelling embeddings."""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

import argparse
import os
from typing import Optional
from ._labels import labels_from_identifiers

import numpy as np
import pandas as pd

from embeddings import LabelledFeatures

# Name for index column
COLUMN_NAME_IDENTIFIER = "identifier"

# Optional placeholder used in image_dir argument
PLACEHOLDER_FOR_SUBSTITUTION = "<IMAGE>"


def load_features(args: argparse.Namespace) -> LabelledFeatures:
    """Loads the embeddings from a CSV file, determines identifiers and labels - all according to the arguments"""

    # Read all columns, text and number
    features = _read_csv(args.file_path_to_csv, encoding=args.encoding)

    # Find the numeric and string columns
    numeric_columns = features.select_dtypes(include=np.number)
    string_columns = features.select_dtypes(include=["object"])

    # Extract or create identifiers for the data-frame
    identifiers = _select_or_create_identifiers(string_columns)

    features_with_identifiers = _add_row_names(numeric_columns.copy(), identifiers)

    # Take the first string col as the row names (index)
    return LabelledFeatures(
        features_with_identifiers,
        _derive_group_label_from_identifiers(features_with_identifiers, args.max_label_index),
        _maybe_image_paths(features_with_identifiers, args.image_dir_path, args.image_dir_sequence),
    )


def _read_csv(file_path_to_csv: str, encoding: str) -> pd.DataFrame:
    """Reads the CSV from the file-system"""
    return pd.read_csv(file_path_to_csv, index_col=None, header=0, encoding=encoding)


def _maybe_image_paths(
    features: pd.DataFrame, image_directory_path: Optional[str], image_directory_sequence: Optional[str]
) -> Optional[pd.Series]:
    """Maybe creates a series of image-paths derived from the index names in data/frame (the returned series has
    identical size and order)

    No paths are created if image_directory_path is None, and instead None is returned.

    :param features: data-frame the images prefer to
    :param image_directory_path: iff present, the index name of df (a relative path) for each feature row is
    appended/substituted to form a complete path to an image
    :param image_directory_sequence: iff present, a six-digit integer sequence for each feature row is
    appended/substituted to form a complete path to an image
    """
    # If neither image_dir argument is set exit
    if (image_directory_path is None) and (image_directory_sequence is None):
        return None

    # If image_dir_path is set, form complete image-paths for each feature-row by using the path
    # (the label in the index) of the data frame to join or substitute
    if image_directory_path:
        return features.index.to_series().map(lambda path: _join_or_substitute(image_directory_path, path))

    # If image_dir_sequence is set, form coomplete image-paths for each feature-row using a six digit sequence to join
    # or substitute
    if image_directory_sequence:
        number_rows = len(features.index)
        sequence = pd.Series(range(0, number_rows))
        return sequence.map(lambda number: _join_or_substitute(image_directory_sequence, "{:06d}".format(number)))


def _join_or_substitute(image_directory: str, path: str) -> str:
    """
    Derives paths to images by either joining path to image_dir or substituting path into image_dir (if it contains
    ``PLACEHOLDER_FOR_SUBSTITUTION``)

    Both paths are normed so that directory-seperators match the execution environment.

    :param image_directory: either the absolute path to a directory OR a such a path with a placeholder
    ``PLACEHOLDER_FOR_SUBSTITUTION`` which can be substituted
    :param path: the relative-path to an image
    :return: either the relative-path joined to image_dir or the relative-path substituted into image_dir in place of
    ``PLACEHOLDER_FOR_SUBSTITUTION``
    """
    if PLACEHOLDER_FOR_SUBSTITUTION in image_directory:
        return os.path.normpath(image_directory).replace(PLACEHOLDER_FOR_SUBSTITUTION, os.path.normpath(path), 1)
    else:
        return os.path.join(image_directory, path)


def _select_or_create_identifiers(string_columns) -> pd.Series:
    """Selects the first (left-most) string column as the identifiers or otherwise creates a range of numbers"""
    if len(string_columns.columns) > 0:
        return string_columns.iloc[:, 0]
    else:
        return pd.Series(range(len(string_columns.columns)))


def _add_row_names(features: pd.DataFrame, row_names: pd.Series) -> pd.DataFrame:
    """Adds a series as row-names to a data-frame"""
    features[COLUMN_NAME_IDENTIFIER] = row_names
    features.set_index(COLUMN_NAME_IDENTIFIER, inplace=True)
    return features


def _derive_group_label_from_identifiers(features: pd.DataFrame, max_label_index: int) -> pd.Series:
    """Derives the first group (leftmost group in name) from the names of a data-frame."""
    return pd.Series(
        list(labels_from_identifiers(features.index.values, max_label_index)), dtype="category", index=features.index
    )

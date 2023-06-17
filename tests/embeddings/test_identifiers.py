"""Tests :mod:`_identifiers`"""
from typing import List, Optional

import pandas as pd
from _data_frame_fixture import ColumnType, create_column, create_data_frame

from anchor_python_visualization.embeddings._identifiers import (
    _create_numeric_sequence, select_or_create_identifiers)

_NUMBER_ROWS: int = 7
"""The number of rows created in each data-frame."""


def test_unique_string():
    """Tests with a data-frame with unique string columns, and several numeric columns."""
    _test_assert(
        ColumnType.UNIQUE_STRING, [ColumnType.UNIQUE_STRING], [ColumnType.UNIQUE_INT]
    )


def test_duplicated_string_without_int():
    """Tests with a data-frame with a duplicated string column, and no numeric columns."""
    _test_assert(None, [ColumnType.WITH_DUPLICATES_STRING], [])


def test_duplicated_string_and_duplicated_int():
    """Tests with a data-frame with a duplicated string column, and no numeric columns."""
    _test_assert(
        None, [ColumnType.WITH_DUPLICATES_STRING], [ColumnType.WITH_DUPLICATES_INT]
    )


def test_duplicated_string_and_unique_int():
    """Tests with a data-frame with a duplicated string column, and no numeric columns."""
    _test_assert(
        ColumnType.UNIQUE_INT,
        [ColumnType.WITH_DUPLICATES_STRING],
        [ColumnType.UNIQUE_INT],
    )


def _test_assert(
    expected_type: Optional[ColumnType],
    string_column_types: List[ColumnType],
    int_column_types: List[ColumnType],
) -> None:
    """Performs the test by creating data-frames with string and int columns and checking the resulting identifiers

    :param expected_type: what we expect the identifiers to look like (converted to str type if necessary). If None,
                          then we expect a numeric sequence.
    :param string_column_types: column types to use in the data-frame for **string**-columns.
    :param int_column_types: column types to use in the data-frame for **int**-columns.
    """
    identifiers = select_or_create_identifiers(
        create_data_frame(_NUMBER_ROWS, *string_column_types),
        create_data_frame(_NUMBER_ROWS, *int_column_types),
    )
    expected_column = _create_column_or_numeric_sequence(expected_type).astype(str)
    assert identifiers.equals(expected_column)


def _create_column_or_numeric_sequence(column_type: Optional[ColumnType]) -> pd.Series:
    """Creates a column of a particular type, or else a numeric sequence if :code:`column_type is None`."""
    if column_type is not None:
        return create_column(_NUMBER_ROWS, column_type)
    else:
        return _create_numeric_sequence(_NUMBER_ROWS)

"""Selects or create unique identifiers for the data-frame."""
import pandas as pd


def select_or_create_identifiers(
    string_columns: pd.DataFrame, numeric_columns: pd.DataFrame
) -> pd.Series:
    """Determines unique identifiers for the data-frame.

    Several approaches are tried, in the following order of priority:
    - The left-most string column (if each value is unique)
    - The left-most numeric column (if each value is unique)
    - A range of numbers from 0..number(rows)

    Selects the first (left-most) string column as the identifiers or otherwise creates a range of numbers.

    The identifiers are always returned as strings.

    :returns: a data-frame with one column, which are unique (string) identifiers.
    """
    if _is_first_column_unique(string_columns):
        return string_columns.iloc[:, 0]
    elif _is_first_column_unique(numeric_columns):
        return numeric_columns.iloc[:, 0].astype(str)
    else:
        number_rows = max(len(string_columns), len(numeric_columns))
        return _create_numeric_sequence(number_rows)


def _create_numeric_sequence(number_rows: int) -> pd.Series:
    """Creates a numeric sequence from 0 (inclusive) to :code:`number_rows` (exclusive)."""
    return pd.Series(map(str, range(number_rows)))


def _is_first_column_unique(columns: pd.DataFrame) -> bool:
    """Determines if the first column exists and is unique.

    :param columns: a data-frame containing zero or more columns.
    :returns: true if the first column exists, and each value in the column is unique.
    """
    return len(columns.columns) > 0 and _has_unique_values(columns.iloc[:, 0])


def _has_unique_values(series: pd.Series) -> bool:
    """Does a series have entirely unique values?"""
    unique_length = len(series.unique())
    return unique_length == len(series)

"""Creates a data-frame with particular types of columns."""
from enum import Enum

import pandas as pd


class ColumnType(Enum):
    """The different types of columns that can be created in a data-frame."""

    UNIQUE_STRING = 1
    """A column with strings, each of which are unique."""

    UNIQUE_INT = 2
    """A column with ints, each of which are unique."""

    WITH_DUPLICATES_STRING = 3
    """A column with strings, which contains duplicates."""

    WITH_DUPLICATES_INT = 4
    """A column with ints, which contains duplicates."""


def create_data_frame(number_rows: int, *args: ColumnType) -> pd.DataFrame:
    """Create a data-frame, with columns of specified types.

    :param number_rows: the number of rows in the data-frame.
    :param args: the respective type for each column.
    :returns: the data-frame.
    """
    dictionary = {
        f"column{key}": create_column(number_rows, value)
        for (key, value) in enumerate(args)
    }
    return pd.DataFrame(dictionary)


def create_column(number_rows: int, column_type: ColumnType) -> pd.Series:
    """Creates a column with either duplicated values or not, and either of string or int type.

    :param number_rows: the number of rows in the data-frame.
    :param column_type: the type of the column.
    :returns: the data-frame.
    """
    if column_type == ColumnType.UNIQUE_STRING:
        return pd.Series(range(number_rows)).astype(str)
    elif column_type == ColumnType.UNIQUE_INT:
        return pd.Series(range(number_rows))
    elif column_type == ColumnType.WITH_DUPLICATES_STRING:
        return pd.Series(["a"] * number_rows)
    elif column_type == ColumnType.WITH_DUPLICATES_INT:
        return pd.Series([2] * number_rows)
    else:
        raise ValueError(f"Unknown column-type: {column_type}")

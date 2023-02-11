"""Routines for loading embeddings from CSV and adding identifiers and labels."""
from __future__ import annotations

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"


import dataclasses
import random
from typing import Optional

import pandas as pd

from .exceptions import InsufficientRowsException


@dataclasses.dataclass(frozen=True)
class LabelledFeatures:
    """Maintains separate data-frames for embeddings and labels, but linked in order and count.

    Both data-frames must have the same number of rows, ordred identically.
    """

    features: pd.DataFrame
    """Data-frame containing only feature-values (numeric) with each row assigned an identifier."""

    labels: pd.Series
    """Series with labels for each item in ``df_features``.

    The series must have the same size and order as ``embeddings``.
    """

    image_paths: Optional[pd.Series] = None
    """Optional series with a path to an image for each item.

    The series must have the same size and order as ``embeddings``.
    """

    def number_items(self) -> int:
        """Returns the number of items (i.e. rows) in the data-frames/series."""
        return len(self.features.index)

    def sample_without_replacement(self, sample_size: int) -> LabelledFeatures:
        """Samples without replacement (taking identical rows from each member data-frame/series).

        Args:
          sample_size: number of items to sample

        Returns:
            a newly created :class:`LabelledFeatures` containing the sample.

        Raises:
          InsufficientRowsException: if there are fewer rows available than :code:`sample_size`.
        """
        number_rows = self.number_items()
        if sample_size > number_rows:
            raise InsufficientRowsException(
                f"Cannot sample {sample_size} rows from a data-frame with only {number_rows} rows"
            )
        elif sample_size == number_rows:
            # Nothing to do
            return self
        else:
            indices = random.sample(range(number_rows), sample_size)
            image_paths = (
                self.image_paths.iloc[indices] if self.image_paths is not None else None
            )
            return LabelledFeatures(
                self.features.iloc[indices, :],
                self.labels.iloc[indices],
                image_paths,
            )

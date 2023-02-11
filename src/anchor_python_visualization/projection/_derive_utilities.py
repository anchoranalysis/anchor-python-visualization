"""Utilities functions common to multiple projection methods."""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

import numpy as np
import pandas as pd


def derive_projected(
    unprojected: pd.DataFrame, projected: np.ndarray, feature_prefix: str
) -> pd.DataFrame:
    """Converts a projected numpy array back into data-frame format with row.names.

    The projected array is derived from a data-frame.

    Args:
        unprojected: the original data-frame from which the projection was derived
        projected: the projection derived from :code:`unprojected`
        feature_prefix: a prefix to use in feature names e.g. :code:`feature_prefix0`,
            :code:`feature_prefix1` etc.

    Returns:
        a data-frame with identical row order and size to :code:`df_orig` but with embeddings from
        the projection.
    """
    num_columns = np.size(projected, 1)
    return pd.DataFrame(
        projected,
        columns=["%s%i" % (feature_prefix, i) for i in range(num_columns)],
        index=unprojected.index,
    )

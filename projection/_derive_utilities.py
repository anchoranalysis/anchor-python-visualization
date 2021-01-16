"""Utilities functions common to multiple projection methods"""
import numpy as np
import pandas as pd


def derive_projected(unprojected: pd.DataFrame, projected: np.array, feature_prefix: str):
    """
    Converts a projected numpy array (derived from a data-frame) back into data-frame format with row.names

    :param unprojected: the original data-frame from which the projection was derived
    :param projected: the projection derived from `unprojected`
    :param feature_prefix: a prefix to use in feature names to become `feature_prefix0`, `feature_prefix1` etc.
    :return: returns a data-frame with identical row order and sze to df_orig but with embeddings from projection
    """
    num_cols = np.size(projected, 1)
    return pd.DataFrame(
        projected,
        columns=['%s%i' % (feature_prefix, i) for i in range(num_cols)],
        index=unprojected.index
    )

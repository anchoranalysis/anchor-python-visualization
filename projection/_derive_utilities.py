"""Utilities functions common to multiple projection methods"""
import numpy as np
import pandas as pd


def derive_projected_df(df_orig: pd.DataFrame, projection: np.array, feature_prefix: str):
    """Converts a projected numpy array (derived from a data-frame) back into data-frame format with row.names"""
    num_cols = np.size(projection,1)
    return pd.DataFrame(
        projection,
        columns=['%s%i' % (feature_prefix,i) for i in range(num_cols)],
        index=df_orig.index
    )

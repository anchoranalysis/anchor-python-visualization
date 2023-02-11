"""T-SNE projection."""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

import pandas as pd
from sklearn.manifold import TSNE

from ._derive_utilities import derive_projected
from ._pca import PCAProjection
from .projector import Projector

MAX_NUMBER_FEATURES_TSNE = 50
"""The maximum number of features allowed before we trigger a PCA beforehand.

This is because T-SNE will not perform well when there are too many features.
"""


PERPLEXITY_TSNE = 30
"""The perplexity used by TSNE unlesss there are too few rows."""


class TSNEProjection(Projector):
    """Projects to two-dimensions using T-SNE

     This is preceded by a PCA projection when ``num(embeddings) > MAX_NUM_FEATURES_TSNE``.

    It produces embeddings TSNE0 and TSNE1.
    """

    # Overriding a base class
    def project(self, features: pd.DataFrame) -> pd.DataFrame:

        # If there are many embeddings, then use PCA first before T-SNE as per recommendation in
        # documentation
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        features = _maybe_project(features)

        perplexity = _calculate_perplexity(features)

        tsne = TSNE(n_components=2, random_state=0, verbose=1, perplexity=perplexity)
        projection = tsne.fit_transform(features)
        # Convert back into a data-frame, assigning feature-names for each component
        return derive_projected(features, projection, "TSNE")


def _maybe_project(features: pd.DataFrame) -> pd.DataFrame:
    """Reduces the number of columns by PCA projection if there are too many."""
    number_columns = len(features.columns)
    if number_columns > MAX_NUMBER_FEATURES_TSNE:
        pca = PCAProjection(2)
        pca.project(features)
    else:
        return features


def _calculate_perplexity(features: pd.DataFrame) -> int:
    """Adjust the perplexity if there are too few values.

    TSNE requires perplexity to be no greater than the number of inputted rows.
    """
    number_rows = len(features.index)
    return min(PERPLEXITY_TSNE, number_rows - 1)

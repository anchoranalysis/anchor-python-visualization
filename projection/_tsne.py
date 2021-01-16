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


class TSNEProjection(Projector):
    """Projects to two-dimensions using T-SNE (preceded by a PCA if num(embeddings) > MAX_NUM_FEATURES_TSNE)

    It produces embeddings TSNE0 and TSNE1.
    """

    def project(self, features: pd.DataFrame) -> pd.DataFrame:

        # If there are many embeddings, then use PCA first before T-SNE as per recommendation in documentation
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        if len(features.columns) > MAX_NUMBER_FEATURES_TSNE:
            pca = PCAProjection(2)
            features = pca.project(features)

        tsne = TSNE(n_components=2, random_state=0, verbose=1)
        projection = tsne.fit_transform(features)
        # Convert back into a data-frame, assigning feature-names for each component
        return derive_projected(features, projection, "TSNE")

"""PCA projection."""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

import pandas as pd
from sklearn import decomposition

from ._derive_utilities import derive_projected
from .projector import Projector


class PCAProjection(Projector):
    """Projects using PCA to a lower dimensional feature-space. Produces embeddings PCA0, PCA1, PCA2 etc."""

    def __init__(self, number_components: int = 2):
        """Constructor

        :param number_components: target number of dimensions for the PCA projection
        """
        self.number_components = number_components

    # Overriding a base class
    def project(self, features: pd.DataFrame) -> pd.DataFrame:

        pca = decomposition.PCA(n_components=self.number_components)
        projection = pca.fit_transform(features)

        print(
            "Total Explained variation: {}".format(pca.explained_variance_ratio_.sum())
        )

        # Convert back into a data-frame, assigning feature-names for each component
        return derive_projected(features, projection, "PCA")

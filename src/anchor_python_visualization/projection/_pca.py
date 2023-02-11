"""PCA projection."""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

import dataclasses

import pandas as pd
from sklearn import decomposition

from ._derive_utilities import derive_projected
from .projector import Projector


@dataclasses.dataclass(frozen=True)
class PCAProjection(Projector):
    """Projects using PCA to a lower dimensional feature-space.

    Produces embeddings PCA0, PCA1, PCA2 etc.
    """

    number_components: int = 2
    """Target number of dimensions for the PCA projection"""

    # Overriding a method in a base class
    def project(self, features: pd.DataFrame) -> pd.DataFrame:

        pca = decomposition.PCA(n_components=self.number_components)
        projection = pca.fit_transform(features)

        print(
            "Total Explained variation: {}".format(pca.explained_variance_ratio_.sum())
        )

        # Convert back into a data-frame, assigning feature-names for each component
        return derive_projected(features, projection, "PCA")

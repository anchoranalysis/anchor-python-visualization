"""PCA projection"""
import pandas as pd
from sklearn import decomposition

from ._derive_utilities import derive_projected
from .projector import Projector


class PCAProjection(Projector):
    """Projects using PCA to a lower dimensional feature-space. Produces embeddings PCA0, PCA1, PCA2 etc."""

    def __init__(self, num_components: int = 2):
        """Constructor

        :param num_components: target number of dimensions for the PCA projection
        """
        self.num_components = num_components

    def project(self, features: pd.DataFrame) -> pd.DataFrame:

        pca = decomposition.PCA(n_components=self.num_components)
        projection = pca.fit_transform(features)

        print("Total Explained variation: {}".format(pca.explained_variance_ratio_.sum()))

        # Convert back into a data-frame, assigning feature-names for each component
        return derive_projected(features, projection, "PCA")

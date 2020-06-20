"""PCA projection"""
import pandas as pd
from sklearn.decomposition import PCA

from ._derive_utilities import derive_projected_df
from .projection import Projection


class PCAProjection(Projection):
    """Projects using PCA to a lower dimensional feature-space. Produces features PCA0, PCA1, PCA2 etc."""
    def __init__(self, num_components: int = 2):
        """Constructor

        Arguments:
        ----------
        num_components:

        """
        self.num_components = num_components

    def project(self, df: pd.DataFrame) -> pd.DataFrame:

        pca = PCA(n_components=self.num_components)
        projection = pca.fit_transform(df)

        print('Total Explained variation: {}'.format(pca.explained_variance_ratio_.sum()))

        # Convert back into a data-frame, assigning feature-names for each component
        return derive_projected_df(df, projection, "PCA")

"""T-SNE projection"""
import pandas as pd
from sklearn.manifold import TSNE

from ._derive_utilities import derive_projected_df
from .pca import PCAProjection
from .projection import Projection

MAX_NUM_FEATURES_TSNE = 50

class TSNEProjection(Projection):
    """Projects to two-dimensions using T-SNE (preceded by a PCA if num(features) > MAX_NUM_FEATURES_TSNE)

    It produces features TSNE0 and TSNE1.
    """

    def project(self, df: pd.DataFrame) -> pd.DataFrame:

        # If there are lots of features, then use PCA first before T-SNE as per recommendation in documentation
        if len(df.columns)>MAX_NUM_FEATURES_TSNE:
            pca = PCAProjection(2)
            df = pca.project(df)

        tsne = TSNE(n_components=2, random_state=0, verbose=1)
        projection = tsne.fit_transform(df)
        # Convert back into a data-frame, assigning feature-names for each component
        return derive_projected_df(df, projection, "TSNE")

from .visualizeScheme import VisualizeScheme
from projection import Projection

from typing import Optional
import pandas as pd
import plotly.express as px
from _labelled_features import LabelledFeatures


class PlotProjection2D(VisualizeScheme):
    """Projects the feature space onto two dimensions and plots"""
    def __init__(self, projection: Projection):
        """Constructor

        Arguments:
        ----------
        projection
            how the projection is performed
        """
        self._projection = projection

    def visualize_data_frame(self, features: LabelledFeatures) -> None:

        df_projected = self._projection.project(features.df_features)

        _plot_first_two_dims_projection(df_projected, features.labels)


def _plot_first_two_dims_projection(df: pd.DataFrame, labels: Optional[pd.Series] = None) -> None:

    # Makes the identifiers a normal column (to see the identifier when hovering over a point in plotly)
    df["identifier"] = df.index

    if labels is not None:
        df["label"] = labels

    fig = px.scatter(
        df,
        x=df.columns[0],
        y=df.columns[1],
        color="label" if labels is not None else None,
        hover_name="identifier"
    )
    fig.show()

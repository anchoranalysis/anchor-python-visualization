from typing import Optional

from projection import Projection
from ._plot_features_projection import PlotFeaturesProjection
from ._tensorboard_export import TensorBoardExport
from .visualize_features_scheme import VisualizeFeaturesScheme

VISUALIZE_FEATURES_DEFAULT_IDENTIFIER = "plot"
VISUALIZE_FEATURES_FACTORY_IDENTIFIERS = [VISUALIZE_FEATURES_DEFAULT_IDENTIFIER, "TensorBoard"]


def create_visualize_features_method(
        method_identifier: Optional[str],
        projection: Optional[Projection],
        output_path: Optional[str]
    ) -> VisualizeFeaturesScheme:
    """
    Creates a visualize-features method from an identifier
    :param method_identifier: string that is one of VISUALIZE_FEATURES_FACTORY_IDENTIFIERS
    :param projection: method for performing projection into smaller dimensionality
    :param output_path: a path for writing any relevant output
    :return: a newly visualize-features
    """
    if method_identifier == "plot" or method_identifier is None:
        return PlotFeaturesProjection(projection)
    elif method_identifier == "TensorBoard":
        return TensorBoardExport(projection, output_path)
    else:
        raise Exception(
            "Unknown identifier for projection: {}".format(method_identifier)
        )
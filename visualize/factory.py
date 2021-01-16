from typing import Optional

from projection import Projection
from ._plot_features_projection import PlotFeaturesProjection
from ._tensorboard_export import TensorBoardExport
from .visualize_features_scheme import VisualizeFeaturesScheme

IDENTIFIERS = ["plot", "TensorBoard"]
DEFAULT_IDENTIFIER = "plot"


def create_method(
                                        method_identifier: Optional[str],
                                        projection: Optional[Projection],
                                        output_path: Optional[str]
                                    ) -> VisualizeFeaturesScheme:
    """
    Creates a visualize-features method from an identifier.

    :param method_identifier: string that is one of :const:`IDENTIFIERS`.
    :param projection: method for performing projection into smaller dimensionality.
    :param output_path: a path for writing any relevant output.
    :returns: a newly created instance corresponding to the identifier.
    """
    if method_identifier == IDENTIFIERS[0] or method_identifier is None:
        return PlotFeaturesProjection(projection)
    elif method_identifier == IDENTIFIERS[1]:
        return TensorBoardExport(projection, output_path)
    else:
        raise Exception(
            "Unknown identifier for projection: {}".format(method_identifier)
        )

"""Creates :class:`~visualize.VisualizeFeaturesScheme`"""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

from typing import Optional

import projection
from ._plot_features_projection import PlotFeaturesProjection
from ._tensorboard_export import TensorBoardExport
from .visualize_features_scheme import VisualizeFeaturesScheme

IDENTIFIERS = ["plot", "TensorBoard"]
DEFAULT_IDENTIFIER = "plot"


def create_method(
    identifier: Optional[str],
    projector: Optional[projection.Projector],
    output_path: Optional[str],
) -> VisualizeFeaturesScheme:
    """
    Creates a visualize-embeddings method from an identifier.

    :param identifier: string that is one of :const:`IDENTIFIERS`.
    :param projector: method for performing projection into smaller dimensionality.
    :param output_path: a path for writing any relevant output.
    :returns: a newly created instance corresponding to the identifier.
    """
    if identifier == IDENTIFIERS[0] or identifier is None:
        return PlotFeaturesProjection(projector)
    elif identifier == IDENTIFIERS[1]:
        return TensorBoardExport(projector, output_path)
    else:
        raise ValueError("Unknown identifier for projection: {}".format(identifier))

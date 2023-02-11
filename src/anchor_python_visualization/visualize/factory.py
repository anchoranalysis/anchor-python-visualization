"""Creates :class:`~visualize.VisualizeFeaturesScheme`"""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

from typing import Optional

from anchor_python_visualization import projection

from ._plot_features_projection import PlotFeaturesProjection
from ._tensorboard_export import TensorBoardExport
from .visualize_features_scheme import VisualizeFeaturesScheme

IDENTIFIERS = ["plot", "tensorboard"]
"""Unique strings to use as command-line-arguments to select a :class:`VisualizeFeaturesScheme`.

All are lower-case.
"""


DEFAULT_IDENTIFIER = "plot"
"""The default choice to use in :const:`IDENTIFIERS`."""


def create_method(
    identifier: Optional[str],
    projector: Optional[projection.Projector],
    output_path: Optional[str],
) -> VisualizeFeaturesScheme:
    """
    Creates a visualize-embeddings method from an identifier.

    Args:
        identifier: string that is one of :const:`IDENTIFIERS`, case-insensitive.
        projector: method for performing projection into smaller dimensionality.
        output_path: a path for writing any relevant output.

    Returns:
        a newly created instance corresponding to the identifier.
    """
    if projector is None:
        raise ValueError("No projector specified.")

    if identifier is None:
        return PlotFeaturesProjection(projector)

    identifier = identifier.casefold()

    if identifier == IDENTIFIERS[0]:
        return PlotFeaturesProjection(projector)
    elif identifier == IDENTIFIERS[1]:

        if output_path is None:
            raise ValueError("An output-path is required but not specified.")

        return TensorBoardExport(projector, output_path)
    else:
        raise ValueError("Unknown identifier for projection: {}".format(identifier))

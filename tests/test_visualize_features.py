"""Tests :mod:`visualize_features`."""
from unittest import mock
from anchor_python_visualization import visualize_features
from _utilities import path_same_directory, call_with_arguments


@mock.patch("matplotlib.pyplot.show")
def test_plot(mock_show: mock.MagicMock) -> None:
    filename = path_same_directory(__file__, "resources/features.csv")
    call_with_arguments(visualize_features, [filename, "-m", "plot"])

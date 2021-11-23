"""Tests :mod:`histogram_plot`."""
from unittest import mock
from anchor_python_visualization import histogram_plot
from _utilities import path_same_directory, call_with_arguments


@mock.patch("matplotlib.pyplot.show")
def test_main(mock_show: mock.MagicMock) -> None:
    filename = path_same_directory(__file__, "resources/histogram.csv")
    call_with_arguments(histogram_plot, [filename])

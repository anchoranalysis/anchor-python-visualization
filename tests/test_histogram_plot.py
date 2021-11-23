"""Tests :mod:`histogram_plot`."""
from unittest import mock
from anchor_python_visualization import histogram_plot
from anchor_python_utilities import file, fixture


@mock.patch("matplotlib.pyplot.show")
def test_main(mock_show: mock.MagicMock) -> None:
    filename = file.path_same_directory(__file__, "resources/histogram.csv")
    fixture.call_with_arguments(histogram_plot, [filename])

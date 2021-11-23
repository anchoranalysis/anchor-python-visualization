"""Tests :mod:`visualize_features`."""
import pathlib
import os
from unittest import mock
from anchor_python_visualization import visualize_features
from anchor_python_utilities import file, fixture
from typing import List


_EXPECTED_TENSORBOARD_FILES: List[str] = [
    "checkpoint",
    "embeddings.ckpt-1.data-00000-of-00001",
    "embeddings.ckpt-1.index",
    "metadata.tsv",
    "projector_config.pbtxt",
]
"""Files expected to be created when the TensorBoard visualization method is used."""


@mock.patch("plotly.express.scatter")
def test_plot(mock_show: mock.MagicMock) -> None:
    """Tests a scatter-plot with Plotly using the default projection method.."""
    _call_wth_features(["-m", "plot"])


def test_tensorboard_pca_with_output_directory(tmp_path: pathlib.Path) -> None:
    """Tests writing TensorBoard files using the PCA projection method, varying case deliberately."""
    _call_wth_features_and_output_path(tmp_path, ["-m", "TensorBoard", "-p", "PcA"])
    _assert_tensorboard_files_exist(tmp_path)


def test_tensorboard_tsne_with_output_directory(tmp_path: pathlib.Path) -> None:
    """Tests writing TensorBoard files using the t-SNE projection method, varying case deliberately."""
    _call_wth_features_and_output_path(tmp_path, ["-m", "tensorBoard", "-p", "t-SNe"])
    _assert_tensorboard_files_exist(tmp_path)


def _call_wth_features_and_output_path(
    tmp_path: pathlib.Path, arguments_additional: List[str]
) -> None:
    """Like :func:`_call_wth_features` but adds a :code:`-o` and the output-path as arguments."""
    _call_wth_features(["-o", str(tmp_path), *arguments_additional])


def _call_wth_features(arguments_additional: List[str]) -> None:
    """Calls the :mod:'visualize_features' module with path to features file in the resources.

    :param arguments_additional: command-line arguments in addition the filename to include in the call.
    """
    filename = file.path_same_directory(__file__, "resources/features.csv")
    fixture.call_with_arguments(visualize_features, [filename, *arguments_additional])


def _assert_tensorboard_files_exist(tmp_path: pathlib.Path) -> None:
    """Asserts particular files have been created as expected for TensorBoard output."""
    for tensorboard_file in _EXPECTED_TENSORBOARD_FILES:
        path = pathlib.Path(os.path.join(tmp_path, tensorboard_file))
        assert path.is_file()

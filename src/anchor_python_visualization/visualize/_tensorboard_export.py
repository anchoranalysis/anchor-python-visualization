"""Exports to TensorBoard"""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

import os
from typing import Optional

import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector

from anchor_python_visualization import embeddings, projection

from ._image_sprite import create_sprite_at
from .visualize_features_scheme import VisualizeFeaturesScheme

# Size of each image in the sprite. let's always keep each element to be an integer divisor of MAX_NUMBER_SAMPLES.
IMAGE_SIZE_IN_SPRITE = (64, 64)

FILENAME_METADATA = "metadata.tsv"
FILENAME_FEATURES = "embeddings.ckpt"
FILENAME_IMAGE_SPRITE = "sprite.png"

# Max sprite size is apparently 8192 x 8192 pixels, so this is the maximum number of images that can
# be supported
MAX_NUMBER_IMAGES_ALLOWED_IN_SPRITE = (8192 // IMAGE_SIZE_IN_SPRITE[0]) * (
    8192 // IMAGE_SIZE_IN_SPRITE[1]
)


class TensorBoardExport(VisualizeFeaturesScheme):
    """Exports embeddings in a format so that TensorBoard can be used for visualization.

    Optionally, image-sprites can be associated with each feature row to be visualized.

    If an image-path is associated with each item, a large tiled image (a sprite) is created with
    small scaled (thumnnail-like) versions of each image. TensorBoard can read this image to show
    the thumbnails alongside data-points.

    If the embeddings have more rows than MAX_NUMBER_IMAGES_ALLOWED_IN_SPRITE then a random-sample
    (without replacement) is taken to reduce the number the embeddings to
    MAX_NUMBER_IMAGES_ALLOWED_IN_SPRITE. Note this introduces non-deterministic behaviour.

    Thanks to the TensorBoard tutorial
    https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin

    Thanks to a medium.com post by Andrew B. Martin for the inspiration
    https://medium.com/looka-engineering/how-to-visualize-feature-vectors-with-sprites-and-tensorflows-tensorboard-3950ca1fb2c7
    """

    def __init__(
        self, projector_method: Optional[projection.Projector], output_path: str
    ):
        """Constructor

        Args:
            projector_method: optional projection to reduce dimensionality before export
            output_path: where to write the "log-dir" for tensorboard
        """
        self._projector = projector_method
        self._output_path = _create_dir_or_throw(output_path)

    # Overriding a base class
    def visualize_data_frame(self, features: embeddings.LabelledFeatures) -> None:

        print("Exporting tensorboard logs to: {}".format(self._output_path))

        features = _sample_if_needed(features)

        path_metadata = self._resolved_path(FILENAME_METADATA)
        path_features = self._resolved_path(FILENAME_FEATURES)

        _write_labels(features.labels, path_metadata)

        _save_embedding_as_checkpoint(
            self._maybe_project(features.features), path_features
        )

        projector_config = _create_projector_config(
            path_metadata, self._maybe_create_sprite(features.image_paths)
        )

        projector.visualize_embeddings(self._output_path, projector_config)

    def _maybe_project(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Apply a projection to the data-frame, if a projection is selected."""
        if self._projector is not None:
            return self._projector.project(data_frame)
        else:
            return data_frame

    def _maybe_create_sprite(self, image_paths: Optional[pd.Series]) -> Optional[str]:
        """Creates a sprite image with thumbnails for many different images, if selected.

        Args:
            image_paths: paths to various images that are populated into the sprite in order

        Returns:
            the path where the sprite was written, or None if no sprite was written.
        """
        if image_paths is not None:
            sprite_path = self._resolved_path(FILENAME_IMAGE_SPRITE)
            create_sprite_at(image_paths, sprite_path, IMAGE_SIZE_IN_SPRITE)
            return sprite_path
        else:
            return None

    def _resolved_path(self, path_relative_to_log_dir: str) -> str:
        """Resolves a path to the log-dir.

        Args:
            path_relative_to_log_dir: a path expressed relative only to the log directory.

        Returns:
            an absolute path.
        """
        return os.path.join(self._output_path, path_relative_to_log_dir)


def _sample_if_needed(
    features: embeddings.LabelledFeatures,
) -> embeddings.LabelledFeatures:
    """Randomly samples if needed.

    This occurs to ensure :code:`num_rows(embeddings) <=MAX_NUMBER_IMAGES_ALLOWED_IN_SPRITE`.
    """
    if features.image_paths is None:
        # Number of rows irrelevant as no sprite will be created, so exit early unchanged
        return features

    num_rows = features.number_items()
    if num_rows > MAX_NUMBER_IMAGES_ALLOWED_IN_SPRITE:
        print(
            f"Sampling {MAX_NUMBER_IMAGES_ALLOWED_IN_SPRITE} rows from a total of {num_rows} rows"
            "in the feature-table as this is the maximum image-sprite"
        )
        return features.sample_without_replacement(MAX_NUMBER_IMAGES_ALLOWED_IN_SPRITE)
    else:
        return features


def _create_projector_config(
    path_metadata: str, path_sprite: Optional[str]
) -> projector.ProjectorConfig:
    """Creates a projector-config as needed to show the embedding in Tensorboard."""
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = path_metadata

    if path_sprite is not None:
        embedding.sprite.image_path = path_sprite
        embedding.sprite.single_image_dim.extend(IMAGE_SIZE_IN_SPRITE)

    return config


def _save_embedding_as_checkpoint(embedding: pd.DataFrame, path: str) -> None:
    """Saves the feature embedding as a checkpointed tensor."""
    weights = tf.Variable(tf.convert_to_tensor(embedding.to_numpy()))
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(path)


def _write_labels(labels: pd.Series, path: str) -> None:
    """Writes each label on a separate line to a file"""
    labels.to_csv(
        path, sep="\t", header=["Label"], index=True, index_label="Identifier"
    )


def _create_dir_or_throw(path: Optional[str]) -> str:
    """Creates a directory if necessary and if defined. If not defined, throw an exception."""
    if path is not None:
        # Make the directory if necessary
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    else:
        raise ValueError("An output-path must be specified for the TensorBoard method")

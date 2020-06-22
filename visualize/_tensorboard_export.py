import os
from typing import Optional

import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector

from features import LabelledFeatures
from projection import Projection
from ._image_sprite import create_sprite_at
from .visualize_features_scheme import VisualizeFeaturesScheme

# Size of each image in the sprite
IMAGE_SIZE_IN_SPRITE = (64, 64)

FILENAME_METADATA = 'metadata.tsv'
FILENAME_FEATURES = 'features.ckpt'
FILENAME_IMAGE_SPRITE = 'sprite.png'


class TensorBoardExport(VisualizeFeaturesScheme):
    """Exports features (and optional image sprites) in a format so that TensorBoard can be used for visualization"

    If an image-path is associated with each item, a large tiled image (a sprite) is created with small scaled
    (thumnnail-like) versions of each image. TensorBoard can read this image to show the thumbnails alongside
    data-points.

    Thanks to the TensorBoard tutorial
    https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin

    Thanks to a medium.com post by Andrew B. Martin for the inspiration
    https://medium.com/looka-engineering/how-to-visualize-feature-vectors-with-sprites-and-tensorflows-tensorboard-3950ca1fb2c7
    """

    def __init__(self, projection: Optional[Projection], output_path: str):
        """Constructor

        :param projection: optional projection to reduce dimensionality before export
        :param output_path: where to write the "log-dir" for tensorboard
        """
        self._projection = projection
        self._output_path = _create_dir_or_throw(output_path)

    def visualize_data_frame(self, features: LabelledFeatures) -> None:

        print("Exporting tensorboard logs to: {}".format(self._output_path))

        path_metadata = self._resolved_path(FILENAME_METADATA)
        path_features = self._resolved_path(FILENAME_FEATURES)

        _write_labels(features.labels, path_metadata)
        _save_embedding_as_checkpoint(
            self._maybe_project(features.df_features),
            path_features
        )

        projector_config = _create_projector_config(
            path_metadata,
            self._maybe_create_sprite(features.image_paths)
        )

        projector.visualize_embeddings(self._output_path, projector_config)

    def _maybe_project(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._projection is not None:
            return self._projection.project(df)
        else:
            return df

    def _maybe_create_sprite(self, image_paths: Optional[pd.Series]) -> Optional[str]:
        if image_paths is not None:
            sprite_path = self._resolved_path(FILENAME_IMAGE_SPRITE)
            create_sprite_at(image_paths, sprite_path, IMAGE_SIZE_IN_SPRITE)
            return sprite_path
        else:
            return None

    def _resolved_path(self, path_relative_to_log_dir: str) -> str:
        """Resolves a path to the log-dir

        :param path_relative_to_log_dir a path expressed relative only to the log dir
        :return an absolute path
        """
        return os.path.join(self._output_path, path_relative_to_log_dir)


def _create_projector_config(path_metadata: str, path_sprite: Optional[str]) -> projector.ProjectorConfig:
    """Creates a projector-config as needed to show the embedding in Tensorboard"""
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = path_metadata

    if path_sprite is not None:
        embedding.sprite.image_path = path_sprite
        embedding.sprite.single_image_dim.extend(IMAGE_SIZE_IN_SPRITE)

    return config


def _save_embedding_as_checkpoint(embedding: pd.DataFrame, path: str) -> None:
    """Saves the feature embedding as a checkpointed tensor"""
    weights = tf.Variable(tf.convert_to_tensor(embedding.to_numpy()))
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(path)


def _write_labels(labels: pd.Series, path: str) -> None:
    """Writes each label on a separate line to a file"""
    labels.to_csv(path, sep="\t", header=["Label"], index=True, index_label="Identifier")


def _create_dir_or_throw(path: Optional[str]) -> str:
    """Creates a directory if necessary and if defined. If not defined, throw an exception."""
    if path is not None:
        # Make the directory if necessary
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    else:
        raise Exception("An output-path must be specified for the TensorBoard method")

"""Creating an image-sprite suitable for TensorBoard from a list of image-paths"""
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd


def create_sprite_at(image_paths: pd.Series, sprite_path: str, image_size_in_sprite: Tuple[int]) -> None:
    """Creates an image-sprite in the format expected by TensorBoard

    This image sprite is a tiled (like a checkboard) version of small identically-sized images (patches) in the order:
      left->right (first)
      top->down (second).

    The sprite MUST always be of square dimensionality, so any unused patches are left blank at the end.

    :param image_paths a series of image-paths for each image that should exist in the sprite (in that order)
    :param sprite_path the path to write the sprite to
    :param image_size_in_sprite the size of each image inside the sprite
    """
    images = []
    for i in range(len(image_paths)):
        path = image_paths[i]

        print("Add image {} of {} to sprite from {}".format(i+1, len(image_paths), path))
        images.append(
            _read_and_scale(path, image_size_in_sprite)
        )

    cv2.imwrite(
        sprite_path,
        _create_sprite(images)
    )


def _read_and_scale(path: str, scale_to_size: Tuple[int]) -> np.array:
    """Reads an image at a path and scales to a particular size"""
    try:
        return cv2.resize(
            _read_with_unicode_path(path),
            scale_to_size
        )
    except (cv2.error, OSError) as err:
        print("An error occurred reading-and-scaling, replacing with an empty thumbnail: {}".format(path))
        print(err)
        return np.zeros((scale_to_size[0],scale_to_size[1],3), np.uint8)


def _read_with_unicode_path(path: str) -> str:
    """
    OpenCV has a problem reading paths which have non-trivial encoding (e.g. unicode). This is a workaround.

    See https://stackoverflow.com/questions/43185605/how-do-i-read-an-image-from-a-path-with-unicode-characters/43185606

    @param path to be opened
    @return an opened image (assuming it is uint8). It will be in BGR format if it is a three channel image.
    """
    return cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED
    )


def _create_sprite(images: List[np.array]) -> np.array:
    """Creates the sprite by tiling into a square image, and adding any padding as needed to complete the square"""
    data = np.array(images)

    data = _convert_grayscale_to_bgr_if_needed(data)

    # Number of images in width and height
    n = _calc_num_images_in_an_axis(data)

    data_padded = _pad_as_needed(data, n)
    return _reshape_into_tiled_square(data_padded, n)


def _calc_num_images_in_an_axis(data: np.array) -> int:
    """Calculates the number of images to place along one axis of the square (i.e. width or height)"""
    return int(np.ceil(
        np.sqrt(data.shape[0])
    ))


def _pad_as_needed(data: np.array, n: int) -> np.array:
    """Adds additional empty images (0 pixels) to ensure there is correct number to complete the square sprite image"""

    # Number of voxels to pad (respectively before and after) in each dimension
    padding = (
        (0, n ** 2 - data.shape[0]),
        (0, 0),
        (0, 0),
        (0, 0)
    )
    return np.pad(
        data,
        padding,
        mode='constant',
        constant_values=0
    )


def _convert_grayscale_to_bgr_if_needed(data: np.array) -> np.array:
    """If data is grayscale then convert it into BGF"""
    if len(data.shape) == 3:
        return np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    else:
        return data


def _reshape_into_tiled_square(data: np.array, n: int):
    """Manipulate the shape of the numpy arrays so it corresponds to one large square 3-channel image"""

    data = data.reshape((n, n) + data.shape[1:])

    data = data.transpose(
        (0, 2, 1, 3, 4)
    )

    return data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:]
    )

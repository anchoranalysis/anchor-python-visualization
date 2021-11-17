r"""A script for visualizing embeddings in a CSV file.

The script:

1. Creates embeddings, by projecting the embeddings into a lower dimensional space.
2. Visualizes the embeddings.

Both steps offer a choice of methods.

---------------
Input Arguments
---------------

Projection methods
------------------

`-p` or `--projection`

 * `t-SNE <https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding>`_ **(default)**
 * `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
 * `none` - unchanged dimensionality for the embeddings.

Visualization methods
----------------------

`-m` or `--method`

 * `plot` - interactive 2D plot of embeddings via `ploty <https://plotly.com/>`_ **(default)**
 * `TensorBoard` - exports a *log directory* to `TensorBoard <https://www.tensorflow.org/tensorboard>`_ at
   `--output-path` or `-o`

Optionally, image thumbnails can be associated with each embedding for `TensorBoard` export with `--image_sequence`
or `--image_path` containing paths where the string :const:`~embeddings.load_features.PLACEHOLDER_FOR_SUBSTITUTION`\
is substituted respectively:

 * with an index from an incrementing six digit integer with leading zeros, corresponding to row order, or,
 * the unique identifier for the embedding.


Structure of the CSV File
-------------------------

The CSV file should have:

   * embeddings as columns.
   * data-items as rows.
   * include headers as the first row.
   * one column called :const:`~embeddings.load_features.COLUMN_NAME_IDENTIFIER` with unique identifiers for each
     embedding.

Otherwise:

  * the *numeric* columns are treated as feature-values
  * the *non-numeric* columns can be combined into a label via the `--max_label_index` argument, combining a number of
    these columns from the left or the right.

Note the label is split into separate groups by a slash (forward or backwards), and `--max_label_index` specifies
a maximum number of groups to be read from the left (if positive) or to be excluded from the right (if negative).

``--encoding`` specifies the encoding of the CSV file as per
`Python's standard encodings <https://docs.python.org/3/library/codecs.html#standard-encodings>`_.

-------------
Example Usage
-------------

Install the package in this repository, by:

* `pip install .` (in the root of the checked out repository) or
* `pip install git+https://github.com/anchoranalysis/anchor-python-visualization.git`

Plotting
--------

Plotting using t-SNE to project to two dimensions.

::

    python -m anchor_python_visualization.visualize_features
        D:\someDirectory\features.csv
        -p t-SNE
        -m plot

TensorBoard export
------------------

1. Create the log-directory:

::

    python -m anchor_python_visualization.visualize_features
        D:\someDirectory\features.csv
        -p none
        -m TensorBoard
        --output D:\someDirectory\tensorboard_logs
        --image_sequence D:\someDirectory\thumbnails\thumbnails_<IMAGE>.png
        -–max_label_index -1

The penultimate parameter is optional, and includes thumbnails.

The ultimate parameter directs the group label to, ignores the "last" port of string i.e. after the final slash.


2. Open the log-directory in `TensorBoard <https://www.tensorflow.org/tensorboard>`_.

::

    tensorboard --logdir D:\someDirectory\tensorboard_logs


3. Open the shown URL, probably `http://localhost:6006/ <http://localhost:6006/>`_

4. Select ``Projector`` from the drop-down list box in the top-right corner.

"""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"


import argparse
import visualize
import projection
import embeddings
from typing import List


def _main():
    """Entry point."""
    args = _arg_parse()

    input_features = embeddings.load_features(args)

    visualize_scheme = visualize.create_method(
        args.method, projection.create_projector(args.projection), args.output
    )
    visualize_scheme.visualize_data_frame(input_features)


def _arg_parse() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize a CSV file with different embeddings."
    )
    parser.add_argument("file_path_to_csv", type=str, help="file-path to a csv file")
    _add_method_via_choices(
        parser,
        "-m",
        "--method",
        visualize.IDENTIFIERS,
        visualize.DEFAULT_IDENTIFIER,
        "visualization",
    )
    _add_method_via_choices(
        parser,
        "-p",
        "--projection",
        projection.IDENTIFIERS,
        projection.DEFAULT_IDENTIFIER,
        "projecting embeddings to smaller dimensionality",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to write any output to for a particular visualization method",
    )
    parser.add_argument(
        "-dp",
        "--image_path",
        help="Identify a directory with thumbnails using the identifier of each image to complete it."
        " If {}  present, instead the identifier is substituted into the path.".format(
            embeddings.PLACEHOLDER_FOR_SUBSTITUTION
        ),
    )
    parser.add_argument(
        "-ds",
        "--image_sequence",
        help="Identify a directory with thumbnails using an incrementing six digit integer"
        " (000000, 000001, 000002 etc.) to substitute for {} in the the path.".format(
            embeddings.PLACEHOLDER_FOR_SUBSTITUTION
        ),
    )
    parser.add_argument(
        "-e",
        "--encoding",
        default=None,
        help="encoding to use when reading the CSV file"
        " (see https://docs.python.org/3/library/codecs.html#standard-encodings for choices)",
    )
    parser.add_argument(
        "-l",
        "--max_label_index",
        default=1,
        type=int,
        help="maximum amount of groups to in include in label leftwards (if positive), or to exclude rightwards"
        " (if negative)",
    )
    return parser.parse_args()


def _add_method_via_choices(
    parser: argparse.ArgumentParser,
    short_name: str,
    long_name: str,
    choices: List[str],
    default_choice: str,
    help_message: str,
) -> None:
    """Adds a multiple-choice method to the parser, case-insensitive."""
    parser.add_argument(
        short_name,
        long_name,
        help="Method to use for {}. Defaults to '{}'".format(
            help_message, default_choice
        ),
        choices=choices,
        default=default_choice,
    )


if __name__ == "__main__":
    _main()

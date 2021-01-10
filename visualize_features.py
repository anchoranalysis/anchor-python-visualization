"""A script for visualizing features in a CSV file in different ways

It first projects the features into an embedding, and then visualizes the embedding using different methods.

Projection methods
------------------
 * t-SNE (default)   https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
 * PCA               https://en.wikipedia.org/wiki/Principal_component_analysis

Visusalization methods
----------------------
 * a 2D interactive plot of points in ploty, opens in the web-browser (default)
 * exports a log-dir which can be opened in TensorBoard

Structure of the CSV File
-------------------------
1. The CSV file should have features as columns, and data-items as rows - and include headers.
2. The numeric columns are treated as feature-values, and the non-numeric columns as labels (handled different depending
on command-line arguments).

Author
------
Owen Feehan
"""
import argparse
from typing import List

from features import load_features, LabelledFeatures
from projection import create_projection_method, PROJECTION_FACTORY_IDENTIFIERS, PROJECTION_FACTORY_DEFAULT_IDENTIFIER
from visualize import (create_visualize_features_method, VISUALIZE_FEATURES_FACTORY_IDENTIFIERS,
                       VISUALIZE_FEATURES_DEFAULT_IDENTIFIER)


def main():
    """Entry point. Expects a path to the CSV file as an argument to the script"""
    args = _arg_parse()

    features: LabelledFeatures = load_features(args)

    visualize_scheme = create_visualize_features_method(
        args.method,
        create_projection_method(args.projection),
        args.output_path,
    )
    visualize_scheme.visualize_data_frame(features)


def _arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualize a CSV file with different features.')
    parser.add_argument('file_path_to_csv', type=str, help='file-path to a csv file')
    _add_method_via_choices(
        parser,
        "-m",
        "--method",
        VISUALIZE_FEATURES_FACTORY_IDENTIFIERS,
        VISUALIZE_FEATURES_DEFAULT_IDENTIFIER,
        "visualization"
    )
    _add_method_via_choices(
        parser,
        "-p",
        "--projection",
        PROJECTION_FACTORY_IDENTIFIERS,
        PROJECTION_FACTORY_DEFAULT_IDENTIFIER,
        "projecting features to smaller dimensionality"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="path to write any output to for a particular visualization method"
    )
    parser.add_argument(
        "-dp",
        "--image_dir_path",
        help="Identify a directory with thumbnails using the identifier of each image to complete it. If {}  present, instead the identifier is substituted into the path."
        .format(PLACEHOLDER_FOR_SUBSTITUTION)
    )
    parser.add_argument(
        "-ds",
        "--image_dir_sequence",
        help="Identify a directory with thumbnails using an incrementing six digit integer (000000, 000001, 000002 etc.) to substitute for {} in the the path."
            .format(PLACEHOLDER_FOR_SUBSTITUTION)
    )
    parser.add_argument(
        "-e",
        "--encoding",
        default=None,
        help="encoding to use when reading the CSV file (see https://docs.python.org/3/library/codecs.html#standard-encodings for choices)"
    )
    return parser.parse_args()


def _add_method_via_choices(
    parser: argparse.ArgumentParser,
    short_name: str,
    long_name: str,
    choices: List[str],
    default_choice: str,
    help_message: str
) -> None:
    """Adds a multiple-choice method to the parser"""
    parser.add_argument(
        short_name,
        long_name,
        help="Method to use for {}. Defaults to '{}'".format(help_message, default_choice),
        choices=choices,
        default=default_choice
    )


if __name__ == "__main__":
    main()
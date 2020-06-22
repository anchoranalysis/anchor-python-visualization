"""A script for visualizing features in a CSV file in different ways

It first projects the features into an embedding, and then visualizes the embedding using different methods.

Projection methods
------------------
 * T-SNE (default)
 * PCA,

Visusalization methods
----------------------
 * a 2D interactive plot of points in ploty, opens in the web-browser (default)
 * exports a log-dir which can be opened in TensorBoard

Structure of the CSV File
-------------------------
1. The CSV file should have features as columns, and data-items as rows - and include headers.
2. The first column is considered to be the name (or other unique identifier like a path) and not a feature.

Grouping of name column
-----------------------
The name can be divided into groups by any forward-slashes (irrespective of operating system)

e.g. a name in the CSV "Europe/Ireland/Dublin"  would form 3 groups of ("Europe","Ireland","Dublin")

This is particular convenient when the name is a path e.g. to an image, to group by (nested) directories.

Author
------
Owen Feehan
"""
import argparse
from typing import List

from features import *
from projection import *
from visualize import *


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
        "-d",
        "--image_dir",
        help="Treat identifier as a path to an image (relative to this dir)"
    )
    return parser.parse_args()


def _add_method_via_choices(
    parser: argparse.ArgumentParser,
    short_name: str,
    long_name: str,
    choices: List[str],
    default_choice: str,
    help_msg: str
) -> None:
    """Adds a multiple-choice method to the parser"""
    parser.add_argument(
        short_name,
        long_name,
        help="Method to use for {}. Defaults to '{}'".format(help_msg,default_choice),
        choices=choices,
        default=default_choice
    )


if __name__ == "__main__":
    main()
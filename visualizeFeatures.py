"""A script for visualizing features in a CSV file in different ways

Specifically:
 1. It applies dimensionality-reduction using PCA
 2. Clusters using T-SNE
 3.  Writes a visualization of the clustering to the filesystem

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
import pandas as pd

from projection import *
from visualize import *
from _labelled_features import LabelledFeatures, load_features


def _arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualize a CSV file with different features.')
    parser.add_argument('file_path_to_csv', type=str, help='file-path to a csv file')
    return parser.parse_args()


def main():
    """Entry point. Expects a path to the CSV file as an argument to the script"""
    args = _arg_parse()

    features: LabelledFeatures = load_features(args)

    visualize_scheme = PlotProjection2D(TSNEProjection())
    visualize_scheme.visualize_data_frame(features)


if __name__ == "__main__":
    main()
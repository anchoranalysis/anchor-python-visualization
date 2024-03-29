r"""Plots a histogram from a CSV file that has `intensity` and `count` in two columns.

---------------
Input Arguments
---------------

* `--file_path_to_csv` a path to the CSV file.

-------------
Example Usage
-------------

Install the package in this repository, by:

* `pip install .` (in the root of the checked out repository) or
* `pip install git+https://github.com/anchoranalysis/anchor-python-visualization.git`

::

    python -m anchor_python_visualization.histogram_plot --file_path_to_csv D:\somedirectory\features.csv

"""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    """Entry point."""
    args = _arg_parse()

    csv = pd.read_csv(args.file_path_to_csv)

    _show_hist(csv["intensity"], csv["count"], 100)


def _arg_parse() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(description="Display a histogram from a CSV.")
    parser.add_argument("file_path_to_csv", type=str, help="file-path to a csv file")
    return parser.parse_args()


def _show_hist(keys: pd.Series, counts: pd.Series, number_bins: int) -> None:
    """Shows a histogram-plot with a logarithmic scale.

    Args:
        keys: a series referring to the keys of the histogram; each key has a corresponding count.
        counts: a series referring to corresponding counts for each key of the histogram, identical
            in size and order to keys.
        number_bins: the number of bins to use in the histogram.
    """
    ax = sns.distplot(
        list(keys),
        hist_kws={"weights": list(counts)},
        norm_hist=False,
        kde=False,
        bins=number_bins,
    )
    ax.set_yscale("log")

    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    main()

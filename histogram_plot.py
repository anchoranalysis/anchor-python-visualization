"""A script for plotting a histogram from a CSV file that has "intensity" and "count" in two columns.

Plots a histogram based upon columns in a CSV file.

Author
-------
Owen Feehan
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Display a histogram from a CSV.')
    parser.add_argument('file_path_to_csv', type=str, help='file-path to a csv file')
    return parser.parse_args()


def _show_hist( keys: pd.Series, counts: pd.Series, num_bins: int ) -> None:
    """Shows a histogram-plot with a logarithmic scale.
    
    Parameters
    -------------
    keys:
        a series referring to the keys of the histogram; each key has a corresponding count
    counts:
        a series referring to corresponding counts for each key of the histogram, identical in size and order to keys
    num_bins:
        the number of bins to use in the histogram
    """
    ax = sns.distplot(
        list(keys),
        hist_kws={"weights": list(counts)},
        norm_hist=False,
        kde=False,
        bins = num_bins
    )
    ax.set_yscale("log")

    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.show()


def main():
    """Entry point. Expects a path to the CSV file as an argument to the script"""
    args = _arg_parse()

    csv = pd.read_csv( args.file_path_to_csv )

    _show_hist(csv['intensity'], csv['count'], 100 )


if __name__ == "__main__":
    main()



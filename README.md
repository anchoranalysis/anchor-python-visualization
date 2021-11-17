# anchor-python-visualization

Scripts in Python for visualizing plots / images etc.

They are installed collectively as a package `anchor_python_visualization`.

## Usage

Each `.py` script in the top-level directory of [src/anchor_python_visualization](https://github.com/anchoranalysis/anchor-python-visualization/tree/master/src/anchor_python_visualization) is designed to be used as a command-line application.

Please first install the package, by:

* `pip install .` (in the root of the checked out repository) or
* `pip install git+https://github.com/anchoranalysis/anchor-python-visualization.git`

A script can then be called from the command-line with the `-m` argument, ala:

```bash
python -m anchor_python_visualization.script_top_level_name --somearg
```

## Scripts

* [visualize_features](https://www.anchoranalysis.org/anchor-python-visualization/autoapi/visualize_features/index.html) ([source](https://github.com/anchoranalysis/anchor-python-visualization/blob/master/src/anchor_python_visualization/visualize_features.py)) - visualizes embeddings in a CSV file in different ways (including exporting to TensorBoard).
* [histogram_plot](https://www.anchoranalysis.org/anchor-python-visualization/autoapi/histogram_plot/index.html) ([source](https://github.com/anchoranalysis/anchor-python-visualization/blob/master/src/anchor_python_visualization/histogram_plot.py)) - plots a histogram from a CSV file.

## Further documentation

* the automatically-generated [API documentation](https://www.anchoranalysis.org/anchor-python-visualization/).
* the comments on the top of each file for more information.
* the [developer guide](https://www.anchoranalysis.org/developer_guide_repositories_anchor_python_visualization.html).

## Author

Owen Feehan

## License

[MIT](LICENSE)
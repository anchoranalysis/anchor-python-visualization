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

See:

* the automatically-generated [API documentation](https://www.anchoranalysis.org/anchor-python-visualization/).
* the comments on the top of each file for more information.
* the [developer guide](https://www.anchoranalysis.org/developer_guide_repositories_anchor_python_visualization.html).

## Author

Owen Feehan

## License

[MIT](LICENSE)
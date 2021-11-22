.. anchor-python-visualization documentation master file, created by
   sphinx-quickstart on Sat Jan 16 14:35:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to anchor-python-visualization's documentation!
=======================================================

Introduction
============

Scripts in Python for visualizing plots / images etc.

They are installed collectively as a package `anchor_python_visualization`.

Usage
=====

Each `.py` script in the top-level directory of `src/anchor_python_visualization <https://github.com/anchoranalysis/anchor-python-visualization/tree/master/src/anchor_python_visualization>`_ is designed as a command-line application.

Please first install the package, by:

* `pip install .` (in the root of the checked out repository) or
* `pip install git+https://github.com/anchoranalysis/anchor-python-visualization.git`

A script can then be called from the command-line with the `-m` argument, ala: ::

   python -m anchor_python_visualization.script_top_level_name --somearg

Top-Level Scripts
=================

- :ref:`histogram_plot <autoapi/histogram_plot/index:Input Arguments>` - plots a histogram from a CSV file (`source <https://github.com/anchoranalysis/anchor-python-visualization/blob/master/src/anchor_python_visualization/histogram_plot.py>`_).
- :ref:`visualize_features <autoapi/visualize_features/index:Introduction>` - visualizes embeddings in a CSV file by plotting or TensorBoard (`source <https://github.com/anchoranalysis/anchor-python-visualization/blob/master/src/anchor_python_visualization/visualize_features.py>`_).

API
===

.. toctree::
   :maxdepth: 4
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

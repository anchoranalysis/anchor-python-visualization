# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys


import os
import sys
import inspect
import shutil

# -- Path setup --------------------------------------------------------------

from anchor_python_sphinx import configure_sphinx

# __location__ = os.path.join(
#    os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe()))
# )

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.join(__location__, "../src"))

# module_dir = os.path.join(__location__, "../src/anchor_python_visualization")


def setup(app):
    configure_sphinx.configure(app)

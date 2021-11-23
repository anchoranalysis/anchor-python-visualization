"""Utility functions to help execute tests."""
import os
import sys
from types import ModuleType
from typing import List


def call_with_arguments(module: ModuleType, arguments: List[str]) -> None:
    """Calls a module with arguments like they were passed on the command-line.

    :param module: a module that must have a :code:`main()` function.
    :param arguments: the arguments that are passed on the command-line (ignoring any initial program-name).
    """
    sys.argv = ["arbitrary_program_name"] + arguments
    module.main()


def path_same_directory(path_file: str, filename: str) -> str:
    """A path to a file that exists in the same directory as :code:`executing_file_path`.

    :param path_file: a path to the file that exists in some directory.
    :param filename: the filename to assign in the outputted path.
    :returns: a path to a file with the directory of :code:`path_file` with `filename` appended.
    """
    return os.path.join(os.path.dirname(path_file), filename)

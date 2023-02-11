"""Labels associated with rows of embeddings, that need not be unique. Useful for showing membership of groups."""

__author__ = "Owen Feehan"
__copyright__ = "Copyright (C) 2021 Owen Feehan"
__license__ = "MIT"
__version__ = "0.1"

from typing import Iterable, List


def labels_from_identifiers(
    identifiers: Iterable[str], max_label_index: int
) -> Iterable[str]:
    """Derives labels from identifiers, by splitting by directory separators.

    Args:
        identifiers: the identifiers.
        max_label_index: maximum amount of groups to in include in label leftwards (if positive), or
        to exclude rightwards (if negative).

    Returns:
        the labels, respectively corresponding to each identifier.
    """

    def extract_groups(identifier: str):
        return _extract_label_from_groups(
            _split_names_into_groups(identifier), max_label_index
        )

    return map(extract_groups, identifiers)


def _extract_label_from_groups(groups: List[str], max_label_index: int) -> str:
    """Derives a label from the groups

    Args:
        groups: the groups an identifier is divided into.
        max_label_index: after splitting the identifier into groups, if positive, this is the
            maximum amount of groups to in include in the label from the left-side, or if negative,
            how many groups to exclude from the right-side.

    Returns:
      the label.
    """
    return "/".join(groups[0:max_label_index])


def _split_names_into_groups(name: str) -> List[str]:
    """Tokenizes a string by slash, either forward or backward."""
    name = name.replace("\\", "/")
    return name.split("/")

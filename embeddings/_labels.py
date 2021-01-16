"""Labels associated with rows of embeddings, that need not be unique. Useful for showing membership of groups."""
from typing import Iterable, List


def labels_from_identifiers(identifiers: Iterable[str], max_label_index: int) -> Iterable[str]:
    """Derives labels from identifiers, by splitting by directory separators.

    :param identifiers: the identifiers
    :param max_label_index:  maximum amount of groups to in include in label leftwards (if positive), or to exclude
    rightwards (if negative)
    :return: the labels, respectively corresponding to each identifier
    """
    def extract_groups(name):
        return _extract_label_from_groups(
            _split_names_into_groups(name),
            max_label_index
        )

    return map(extract_groups, identifiers)


def _extract_label_from_groups(groups: List[str], max_label_index: int) -> str:
    """Derives a label from the groups

    :param groups: the groups an identifier is divided into.
    :param max_label_index: after splitting the identifier into groups, if positive, this is the maximum amount of
           groups to in include in the label from the left-side, or if negative, how many groups to exclude from
           the right-side.
    :return: the label
    """
    return "/".join(groups[0:max_label_index])


def _split_names_into_groups(name: str) -> List[str]:
    """Tokenizes a string by slash, either forward or backward"""
    name = name.replace("\\", "/")
    return name.split("/")

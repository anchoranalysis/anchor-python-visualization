"""Methods for loading embeddings and determining labels"""
from .label import LabelledFeatures
from .load_features import load_features, PLACEHOLDER_FOR_SUBSTITUTION, COLUMN_NAME_IDENTIFIER
from .exceptions import InsufficientRowsException

from projection import create_projector, IDENTIFIERS, Projector
import pandas as pd
import numpy as np


_DATA_FRAME_SIZE = (100, 4)


def test_create_projector() -> None:
    """Tests every projection method that is not None"""

    data_frame = _create_data_frame()

    for identifier in IDENTIFIERS:
        projection = create_projector(identifier)

        if projection is not None:
            _test_projector(projection, data_frame, identifier)


def _test_projector(projection: Projector, data_frame, identifier: str):
    smaller = projection.project(data_frame)
    assert smaller.shape[0] == data_frame.shape[0], identifier + " number of rows equal"
    assert smaller.shape[1] < data_frame.shape[1], identifier + " number of columns smaller"


def _create_data_frame() -> pd.DataFrame:
    return pd.DataFrame(np.random.randint(0, 100, size=_DATA_FRAME_SIZE), columns=list("ABCD"))

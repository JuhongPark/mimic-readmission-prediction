import numpy as np
from src.discretizer import StandardDiscretizer


def test_channel_count():
    d = StandardDiscretizer()
    assert len(d._id_to_channel) == 17


def test_impute_strategies():
    for strategy in ['zero', 'normal_value', 'previous', 'next']:
        d = StandardDiscretizer(impute_strategy=strategy)
        assert d._impute_strategy == strategy


def test_transform_end_t_hours_output():
    d = StandardDiscretizer(timestep=1.0, store_masks=True, impute_strategy='zero')

    # One Heart Rate reading at t=0.5 (index 25 in _ORI_HEADER)
    row = [""] * 65
    row[0] = "0.5"
    row[25] = "80"

    data, header = d.transform_end_t_hours([row], los=2.0)

    # 2 time bins (los=2.0, timestep=1.0)
    # 59 data columns + 17 mask columns = 76
    assert data.shape == (2, 76)
    # Heart Rate mask (channel 8) set in first bin only
    assert data[0, 59 + 8] == 1.0
    assert data[1, 59 + 8] == 0.0
    # Heart Rate value at data column 50
    assert data[0, 50] == 80.0

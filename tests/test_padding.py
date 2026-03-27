import numpy as np
from src.data.padding import pad_zeros


def test_pad_to_max_length():
    arr = [np.ones((10, 4)), np.ones((5, 4))]
    result = pad_zeros(arr, max_len=48)
    assert result.shape == (2, 48, 4)
    assert np.all(result[0, :10] == 1.0)
    assert np.all(result[0, 10:] == 0.0)


def test_min_length_extends_padding():
    arr = [np.ones((10, 4))]
    result = pad_zeros(arr, min_length=60, max_len=48)
    assert result.shape[1] == 60


def test_dtype_preserved():
    arr = [np.ones((5, 3), dtype=np.float32)]
    result = pad_zeros(arr, max_len=48)
    assert result.dtype == np.float32

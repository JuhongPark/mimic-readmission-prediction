from __future__ import annotations
import numpy as np


def pad_zeros(arr: list[np.ndarray], min_length: int | None = None, max_len: int = 48) -> np.ndarray:
    dtype = arr[0].dtype
    ret = [
        np.concatenate(
            [x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)],
            axis=0,
        )
        for x in arr
    ]
    if min_length is not None and ret[0].shape[0] < min_length:
        ret = [
            np.concatenate(
                [x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)],
                axis=0,
            )
            for x in ret
        ]
    return np.array(ret)

from numbers import Number
from typing import Type, Union

import numba as nb
import numpy as np
from numba import cuda

T = Type[Union[Number, np.number]]


@nb.njit(parallel=True)
def cp_tril(x, y):
    """Copy the lower half of the square matrix x into a linear y"""
    k = 0
    for i in nb.prange(x.shape[0]):
        for j in range(i + 1):
            y[k] = x[i, j]
            k += 1


class triu_matrix:
    def __init__(self, arr):
        assert arr.shape[0] == arr.shape[1]
        assert arr.ndim == 2

        self.shape = arr.shape
        self.ndim = arr.ndim

        self.data = np.empty((self.shape[0] + 1) * self.shape[0] // 2, dtype=arr.dtype)
        cp_tril(arr, self.data)

    def _idx(self, i, j):
        return i * self.shape[0] + j

    def __len__(self):
        return self.shape[0]

    def _row(self, i):
        return self.data[i * self.shape[0] : i * self.shape[0] + i]

    def _col(self, j):
        return self.data[j : self.shape[0] * self.shape[0] : self.shape[0]]

    def __repr__(self) -> str:
        return repr(self.data)

    def __str__(self) -> str:
        return str(self.data)

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            i = index[0]
            j = index[1]
            self.data[i * self.shape[0] + j] = value
        else:
            i = index
            self.data[i * self.shape[0] : i * self.shape[0] + i] = value

    def __getitem__(self, index) -> Union[np.ndarray, T]:
        if isinstance(index, tuple):
            i = index[0]
            j = index[1]
            return self.data[i * self.shape[0] + j]
        elif isinstance(index, int):
            i = index
            return self._row(i)

        else:
            raise TypeError("Index must be int or tuple")


if __name__ == "__main__":
    a = np.arange(9).reshape(3, 3)
    b = triu_matrix(a)
    print(b)
    print(b[0])
    print(b[1])

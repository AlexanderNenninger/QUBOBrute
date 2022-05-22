from typing import Callable, Dict, Tuple, Union

import numba as nb
import numpy as np
from numba import cuda

Qubo = Dict[Tuple[int, int], float]


def to_mat(qubo: Qubo) -> np.ndarray:
    """Turns a QUBO dictionary representation into
    a matrix representation.

    Args:
        qubo (Tuple[Qubo, float]): QUBO as returned by PyQubo

    Raises:
        ValueError: If an empty QUBO is provided.

    Returns:
        np.ndarray: QUBO as numpy matrix
    """
    if len(qubo) > 0:
        matsize = max(max(qubo)) + 1
    else:
        raise ValueError("Provide at least some data.")

    w = np.zeros((matsize, matsize), dtype=np.float32)
    for idx in qubo:
        w[idx[0], idx[1]] = qubo[idx]

    return w


@nb.njit(fastmath=True)
def bits(n: Union[int, np.intp], nbits: int) -> np.ndarray:
    """Turn n into an array of float32.

    Args:
        n (int)
        nbits (int): length of output array

    Returns:
        The bits of n in an array of float32
    """
    bits = np.zeros(nbits, dtype=np.float32)
    i = 0
    while n > 0:
        n, rem = n // 2, n % 2
        bits[i] = rem
        i += 1
    return bits


@nb.njit(parallel=True, fastmath=True)
def solve_cpu(Q, c):
    """Calculate all possible values of the QUBO H(x) = x^T Q x + c in parallel on the CPU.

    Args:
        Q (np.ndarray)
        c (float32)

    Returns:
        np.ndarray: all possible values H can take.
    """
    nbits = Q.shape[0]
    N = 2**nbits
    out = np.zeros(N, dtype=np.float32)
    for i in nb.prange(N):
        xs = bits(i, nbits)
        out[i] = xs @ Q @ xs + c

    return out


# CUDA Code starts here
@cuda.jit(device=True)
def cu_bits(n, xs):
    i = 0
    while n > 0:
        n, rem = n // 2, n % 2
        xs[i] = rem
        i += 1


@cuda.jit(device=True)
def cu_qnorm(q, x):
    """Calculate x^T q x inside the CUDA kernel

    Args:
        q (_type_): 2d array of size nbits x nbits
        x (_type_): 1d array of size nbits

    Returns:
        float: x^T q x
    """
    n = q.shape[0]
    out = 0
    for i in range(n):
        tmp = 0
        for j in range(n):
            tmp += q[i, j] * x[j]
        out += tmp * x[i]

    return out


@cuda.jit(device=True)
def copy_slice(a, b, start, end):
    """Copy values inside a kernel from one array to another.

    Args:
        a: source array
        b: target array
        start (_type_): start index
        end (_type_): end index
    """
    for i in range(start, end):
        b[i] = a[i]


def solve_gpu(Q: np.ndarray, c: np.float32) -> np.ndarray:
    """Solve QUBO H(x) = x^T Q x + c on a GPU.

    Args:
        q (np.ndarray): Q
        c (np.float32): energy offset

    Returns:
        v (np.ndarray): All possible energy values H can take in enumerated order.
        Suppose  argmin(v) = i, then bits(i, q.shape[0]) minimizes H.
    """
    assert (
        Q.ndim == 2
    ), f"q needs to be a square matrix. Got {Q.ndim=}, but expected q.ndim=2."
    assert Q.shape[0] == Q.shape[1], "q needs to be a square matrix."

    nbits = Q.shape[0]
    N = 2**nbits

    @cuda.jit()
    def kernel(q, c, solutions):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x
        idx = tx + ty * bw  # type: ignore
        xs = cuda.local.array(nbits, dtype=nb.u1)  # type: ignore
        cu_bits(idx, xs)
        if 0 <= idx < solutions.size:
            solutions[idx] = cu_qnorm(q, xs) + c

    solutions = cuda.device_array(N, dtype=np.float16)
    threadsperblock = 256
    blockspergrid = (solutions.size + (threadsperblock - 1)) // threadsperblock

    Q = cuda.to_device(Q)
    kernel[blockspergrid, threadsperblock](Q, c, solutions)  # type: ignore

    return solutions.copy_to_host()

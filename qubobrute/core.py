from asyncio import Queue
from typing import Callable, Dict, Tuple

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


@nb.njit
def bits(n: int, nbits: int):
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


@nb.njit(parallel=True)
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
def cubits(n, xs):
    i = 0
    while n > 0:
        n, rem = n // 2, n % 2
        xs[i] = rem
        i += 1


@cuda.jit(device=True)
def qnorm(q, x):
    n = q.shape[0]
    out = 0
    for i in range(n):
        tmp = 0
        for j in range(n):
            tmp += q[i, j] * x[j]
        out += tmp * x[i]

    return out


def solve_gpu(Q: np.ndarray, c: np.float32) -> np.ndarray:
    """Solve QUBO H(x) = x^T Q x + c on a GPU.

    Args:
        q (np.ndarray): Q
        c (np.float32): energy offset

    Returns:
        v (np.ndarray): Alll possible values, H can take in enumerated order. Suppose  argmin(v) = i, then bits(i, q.shape[0]) minimizes H.
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
        xs = cuda.local.array(nbits, dtype=nb.u1)
        cubits(idx, xs)
        if idx < solutions.size:
            solutions[idx] = qnorm(q, xs) + c

    solutions = cuda.device_array(N, dtype=np.float16)
    threadsperblock = 256
    blockspergrid = (solutions.size + (threadsperblock - 1)) // threadsperblock
    kernel[blockspergrid, threadsperblock](Q, c, solutions)

    return solutions.copy_to_host()

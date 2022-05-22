import warnings
from math import exp
from typing import Tuple

import cupy as cp
import numba as nb
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from .core import copy_slice, cu_qnorm


@nb.njit(fastmath=True)
def rand_bits(size: int) -> np.ndarray:
    """Generate a vector of random f32 bits.

    Args:
        size (int): number of bits to generate.

    Returns:
        np.ndarray: random array of bits.
    """
    arr = np.zeros(size, dtype=np.float32)
    for i in range(size):
        arr[i] = np.round(np.random.rand())

    return arr


@nb.njit(parallel=True, fastmath=True)
def simulate_annealing(
    q: np.ndarray,
    c: float,
    n_iter: int,
    n_samples: int,
    temperature: float,
    cooling_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate annealing on the CPU.
    Args:
        q (np.ndarray): QUBO as numpy matrix
        c (float): offset of the QUBO
        n_iter (int): number of iterations
        n_samples (int): number of samples to take
        temperature (float): initial temperature
        cooling_rate (float): cooling rate
    Returns:
        Tuple[np.ndarray, np.ndarray]: energies, samples
    """
    n = q.shape[0]
    q = q.astype(np.float32)
    samples = np.zeros((n_samples, n), dtype=np.int8)
    energies = np.zeros(n_samples, dtype=np.float32)

    for i in nb.prange(n_samples):  # type: ignore # noqa
        temperature_local = temperature
        sample = rand_bits(n)
        energy = np.dot(sample, np.dot(q, sample)) + c
        best = sample

        for _ in range(n_iter):
            idx = np.random.randint(0, n)
            new_sample = sample.copy()
            new_sample[idx] = not new_sample[idx]

            new_energy = np.dot(sample, np.dot(q, sample)) + c
            if new_energy < energy:
                sample = new_sample
                energy = new_energy
                best = sample
            else:
                if np.random.rand() < np.exp(
                    min(-(new_energy - energy) / temperature, 0)
                ):
                    sample = new_sample
                    energy = new_energy
            temperature_local *= cooling_rate
            temperature_local = max(temperature, 1e-6)

        samples[i] = best
        energies[i] = best @ q @ best + c

    return energies, samples


def simulate_annealing_gpu(
    q: np.ndarray,
    c: float,
    n_iter: int,
    n_samples: int,
    temperature: float,
    cooling_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate annealing on the GPU. This functions works very well if the QUBO is small,
    but good solutions are required.

    Args:
        q (np.ndarray): QUBO as numpy matrix
        c (float): offset of the QUBO
        n_iter (int): number of iterations
        n_samples (int): number of samples to take
        temperature (float): initial temperature
        cooling_rate (float): cooling rate
    Returns:
        Tuple[np.ndarray, np.ndarray]: energies, samples
    """

    n = q.shape[0]

    @cuda.jit(fastmath=True)
    def simulate_annealing_kernel(rng_states, samples, energies, Q, temperatures):
        """Run one simulated annealing cycle on the GPU. This functions parallelizes over
        the samples, thus lending itself to small qubos and large sample counts.

        Args:
            rng_states (rng_states): states of the random number generator
            samples (device_array): array of samples to anneal
            energies (device_array): energies of the samples
            Q (device_array): QUBO as 2d device array
            temperatures (device_array): temperatures to anneal at
        """
        sample_id = cuda.grid(1)  # type: ignore # noqa
        if sample_id < n_samples:  # type: ignore

            for _ in range(n_iter):
                new_sample = cuda.local.array(n, dtype=np.float32)  # type: ignore
                copy_slice(samples[sample_id], new_sample, 0, n)

                rand_idx_f = xoroshiro128p_uniform_float32(rng_states, sample_id)
                rand_idx = int(rand_idx_f * n)  # type: ignore

                new_sample[rand_idx] = not new_sample[rand_idx]  # type: ignore

                new_energy = cu_qnorm(Q, new_sample) + c

                if new_energy < energies[sample_id]:
                    copy_slice(new_sample, samples[sample_id], 0, n)
                    energies[sample_id] = new_energy
                else:
                    rand_f = xoroshiro128p_uniform_float32(rng_states, sample_id)
                    if rand_f < exp(  # type: ignore
                        min(
                            -(new_energy - energies[sample_id])
                            / temperatures[sample_id],
                            0,
                        )
                    ):
                        copy_slice(new_sample, samples[sample_id], 0, n)
                        energies[sample_id] = new_energy

                temperatures[sample_id] *= cooling_rate
                temperatures[sample_id] = max(temperatures[sample_id], 1e-6)

    samples = np.round(np.random.rand(n_samples, n).astype(np.float32))
    samples = cuda.to_device(samples)

    energies = np.full(n_samples, np.inf, dtype=np.float32)
    energies_cu = cuda.to_device(energies)

    q = q.astype(np.float32)
    q = cuda.to_device(q)

    # Numba complains about inefficient Grid Size. But it is not a problem.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temperatures = cuda.device_array(n_samples, dtype=np.float32)
        temperatures[:] = temperature

    # Saturate GPU SMs but don't overflow thread count
    threadsperblock = min(
        n_samples // 2 // cuda.get_current_device().MULTIPROCESSOR_COUNT, 256
    )

    blockspergrid = (n_samples + (threadsperblock - 1)) // threadsperblock

    rng_states = create_xoroshiro128p_states(blockspergrid * threadsperblock, seed=0)
    simulate_annealing_kernel[blockspergrid, threadsperblock](rng_states, samples, energies_cu, q, temperatures)  # type: ignore

    energies = energies_cu.copy_to_host()
    assert isinstance(energies, np.ndarray)

    samples = samples.copy_to_host()
    assert isinstance(samples, np.ndarray)

    return energies, samples


@nb.jit(forceobj=True, parallel=True)
def simulate_annealing_large_gpu(
    q: np.ndarray,
    c: float,
    n_iter: int,
    n_samples: int,
    temperature: float,
    cooling_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate annealing on the GPU. This functions is parallelized over the entries of q and thus works very well if the QUBO is large, but the number of samples is small.

    Args:
        rng_states (rng_states): states of the random number generator
        samples (device_array): array of samples to anneal
        energies (device_array): energies of the samples
        q (device_array): QUBO as 2d device array
        temperatures (device_array): temperatures to anneal at
    """
    n = q.shape[0]
    q = cp.asarray(q, dtype=cp.float32)
    samples = cp.zeros((n_samples, n), dtype=cp.float32)  # type: ignore
    energies = cp.full(n_samples, np.inf, dtype=cp.float32)

    for i in nb.prange(n_samples):  # type: ignore # noqa
        temperature_local = cp.empty(1, dtype=cp.float32)  # type: ignore
        temperature_local[0] = temperature

        c_local = cp.empty(1, dtype=cp.float32)  # type: ignore
        c_local[0] = c

        samples[i] = cp.round(cp.random.rand(n))
        energies[i] = samples[i] @ q @ samples[i] + c_local
        best = samples[i]

        for _ in range(n_iter):
            idx = cp.random.randint(n)
            new_sample = samples[i].copy()
            new_sample[idx] = not new_sample[idx]
            new_energy = new_sample @ q @ new_sample + c_local

            if new_energy < energies[i]:
                best = new_sample
                energies[i] = new_energy
                samples[i] = new_sample
            else:
                if cp.random.rand() < cp.exp(  # type: ignore
                    -(new_energy - energies[i]) / temperature
                ):
                    samples[i] = new_sample
                    energies[i] = new_energy

            temperature_local *= cooling_rate
            temperature_local = max(temperature_local, 1e-6)

        samples[i] = best
        energies[i] = best @ q @ best + c

    return energies.get(), samples.get()

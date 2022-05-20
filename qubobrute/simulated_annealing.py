import warnings
from math import exp
from typing import Tuple

import numba as nb
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from .core import copy_slice, qnorm


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
    Q: np.ndarray,
    c: float,
    n_iter: int,
    n_samples: int,
    temperature: float,
    cooling_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate annealing on the CPU.
    Args:
        Q (np.ndarray): QUBO as numpy matrix
        c (float): offset of the QUBO
        n_iter (int): number of iterations
        n_samples (int): number of samples to take
        temperature (float): initial temperature
        cooling_rate (float): cooling rate
    Returns:
        Tuple[np.ndarray, np.ndarray]: energies, samples
    """
    n = Q.shape[0]
    samples = np.zeros((n_samples, n), dtype=np.int8)
    energies = np.zeros(n_samples, dtype=np.float32)
    for i in nb.prange(n_samples):
        temperature_local = temperature
        sample = rand_bits(n)
        energy = np.dot(sample, np.dot(Q, sample)) + c
        best = sample

        for j in range(n_iter):
            idx = np.random.randint(0, n)
            new_sample = sample.copy()
            new_sample[idx] = not new_sample[idx]

            new_energy = np.dot(sample, np.dot(Q, sample)) + c
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
        energies[i] = energy

    return energies, samples


def simulate_annealing_gpu(
    Q: np.ndarray,
    c: float,
    n_iter: int,
    n_samples: int,
    temperature: float,
    cooling_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate annealing on the GPU. This functions works very well, if the QUBO is small,
    but good solutions are required.

    Args:
        Q (np.ndarray): QUBO as numpy matrix
        c (float): offset of the QUBO
        n_iter (int): number of iterations
        n_samples (int): number of samples to take
        temperature (float): initial temperature
        cooling_rate (float): cooling rate
    Returns:
        Tuple[np.ndarray, np.ndarray]: energies, samples
    """

    n = Q.shape[0]

    @cuda.jit
    def simulate_annealing_kernel(rng_states, samples, energies, Q, temperatures):
        """Run one simulated annealing cycle on the GPU. Since this is a
        CUDA kernel, we can schedule a massive number of threads.

        Args:
            rng_states (rng_states): states of the random number generator
            samples (device_array): array of samples to anneal
            energies (device_array): energies of the samples
            Q (device_array): QUBO as 2d device array
            temperatures (device_array): temperatures to anneal at
        """
        sample_id = cuda.grid(1)  # type: ignore
        if sample_id < n_samples:  # type: ignore

            for i in range(n_iter):
                new_sample = cuda.local.array(n, dtype=np.float32)  # type: ignore
                copy_slice(samples[sample_id], new_sample, 0, n)

                rand_idx_f = xoroshiro128p_uniform_float32(rng_states, sample_id)
                rand_idx = int(rand_idx_f * n)  # type: ignore

                new_sample[rand_idx] = not new_sample[rand_idx]  # type: ignore

                new_energy = qnorm(Q, new_sample) + c

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
    energies = cuda.to_device(energies)

    Q = Q.astype(np.float32)
    Q = cuda.to_device(Q)

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
    simulate_annealing_kernel[blockspergrid, threadsperblock](rng_states, samples, energies, Q, temperatures)  # type: ignore

    return energies.copy_to_host(), samples.copy_to_host()  # type: ignore

from typing import Tuple

import numba as nb
import numpy as np


@nb.njit
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


@nb.njit(parallel=True)
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
        Tuple[np.ndarray, np.ndarray]: samples
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

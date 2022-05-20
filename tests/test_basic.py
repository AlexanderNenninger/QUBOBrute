import unittest
import warnings

import numpy as np
from pyqubo import Array, Spin
from qubobrute import *


class BasicTest(unittest.TestCase):

    EPSILON = 1e-10

    def setUp(self) -> None:
        s1, s2, s3, s4 = Spin("s1"), Spin("s2"), Spin("s3"), Spin("s4")
        H = (4 * s1 + 2 * s2 + 7 * s3 + s4) ** 2
        self.model = H.compile()
        self.qubo, self.offset = self.model.to_qubo(index_label=True)
        self.q = to_mat(self.qubo)

        self.true_solution = np.array(
            [
                196.0,
                36.0,
                100.0,
                4.0,
                0.0,
                64.0,
                16.0,
                144.0,
                144.0,
                16.0,
                64.0,
                0.0,
                4.0,
                100.0,
                36.0,
                196.0,
            ],
            dtype=np.float32,
        )

    def test_basic(self):
        qubo, offset = self.model.to_qubo(index_label=True)
        q = to_mat(qubo)

        truth = np.array(
            [
                [-160.0, 64.0, 224.0, 32.0],
                [0.0, -96.0, 112.0, 16.0],
                [0.0, 0.0, -196.0, 56.0],
                [0.0, 0.0, 0.0, -52.0],
            ]
        )

        self.assertTrue((truth == q).all())

    def test_cpu(self):
        solutions = solve_cpu(self.q, self.offset)
        self.assertTrue(np.allclose(solutions, self.true_solution))

    def test_solve(self):
        solutions = solve_gpu(self.q, self.offset)
        self.assertTrue(np.allclose(solutions, self.true_solution))

    def test_large(self):
        nbits = 25
        x = Array.create("x", shape=(nbits,), vartype="BINARY")
        H = (np.arange(nbits) @ x - 12) ** 2
        model = H.compile()
        qubo, offset = model.to_qubo(index_label=True)
        q = to_mat(qubo)
        energies = solve_gpu(q, offset)

        self.assertAlmostEqual(int(np.log2(len(energies))), nbits)
        self.assertAlmostEqual(energies.min(), 0)


class TestSimulatedAnnealing(unittest.TestCase):
    def setUp(self) -> None:
        s1, s2, s3, s4 = Spin("s1"), Spin("s2"), Spin("s3"), Spin("s4")
        H = (4 * s1 + 2 * s2 + 7 * s3 + s4) ** 2
        self.model = H.compile()
        self.qubo, self.offset = self.model.to_qubo(index_label=True)
        self.q = to_mat(self.qubo)

        self.true_solution = np.array(
            [
                196.0,
                36.0,
                100.0,
                4.0,
                0.0,
                64.0,
                16.0,
                144.0,
                144.0,
                16.0,
                64.0,
                0.0,
                4.0,
                100.0,
                36.0,
                196.0,
            ],
            dtype=np.float32,
        )

    def test_simulated_annealing(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            energies, solutions = simulate_annealing(
                self.q,
                self.offset,
                n_iter=100,
                n_samples=100,
                temperature=1,
                cooling_rate=0.99,
            )
        min_energy = np.min(energies)
        best = solutions[energies.argmin()]
        self.assertAlmostEqual(min_energy, 0.0)

    def test_cuda_available(self):
        from numba import cuda

        self.assertTrue(cuda.is_available())

    def test_simulated_annealing_gpu(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            energies, solutions = simulate_annealing_gpu(
                self.q,
                self.offset,
                n_iter=1000,
                n_samples=100_000,
                temperature=10,
                cooling_rate=0.99,
            )

        min_energy = np.min(energies)
        best = solutions[energies.argmin()]
        self.assertAlmostEqual(min_energy, 0.0)

    def test_large(self):
        nbits = 1_000
        q = np.random.standard_normal((nbits, nbits))
        offset = 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            energies, solutions = simulate_annealing_gpu(
                q,
                offset,
                n_iter=100,
                n_samples=100_000,
                temperature=10,
                cooling_rate=0.99,
            )

        min_energy = np.min(energies)
        best = solutions[energies.argmin()]
        self.assertLess(min_energy, np.inf)


if __name__ == "__main__":
    unittest.main()

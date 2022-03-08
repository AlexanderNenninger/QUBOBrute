import unittest

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
        self.q = to_mat((self.qubo, self.offset))

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
        qubo = self.model.to_qubo(index_label=True)
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
        nbits = 30
        x = Array.create("x", shape=(nbits,), vartype="BINARY")
        H = (np.arange(nbits) @ x - 12) ** 2
        model = H.compile()
        qubo, offset = model.to_qubo(index_label=True)
        q = to_mat((qubo, offset))
        solutions = solve_gpu(q, offset)

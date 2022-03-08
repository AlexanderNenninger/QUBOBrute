# QUBOBrute - GPU Accelerated Brute Force Solver for Quadratic Binary Optimization Problems

## Prerequisits

This code is tested on an NVIDIA RTX 2060 running on Ubuntu 20.04 LTS. With 6GB of VRAM, the GPU can solve qubos of up to 30 variables.

## Installation

    `pip install -r requirements.txt`

## Usage

    from qubobrute.core import *
    from pyqubo import Array

    nbits = 30
    x = Array.create("x", shape=(nbits,), vartype="BINARY")
    H = (np.arange(nbits) @ x - 12) ** 2
    model = H.compile()
    qubo, offset = model.to_qubo(index_label=True)
    q = to_mat((qubo, offset))
    solutions = solve_gpu(q, offset)

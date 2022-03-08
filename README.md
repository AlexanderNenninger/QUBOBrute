# QUBOBrute

*GPU Accelerated Brute Force Solver for Quadratic Binary Optimization Problems.*

-----------------

The goal of this project is to provide a simple, extensible GPU accelerated QUBO Solver for fast debugging of problem definitions.

## Prerequisits

This project is tested on an NVIDIA RTX 2060 running on Ubuntu 20.04 LTS. With 6GB of VRAM, the GPU can solve qubos of up to 30 variables.

## Installation

`conda install --file condarequirements.txt`

In particular, you need

+ `numba`
+ `pyqubo`
+ `cudatoolkit`
+ `scipy`

## Usage

    from qubobrute import *
    from pyqubo import Array

    nbits = 30
    x = Array.create("x", shape=(nbits,), vartype="BINARY")
    H = (np.arange(nbits) @ x - 12) ** 2
    model = H.compile()
    qubo, offset = model.to_qubo(index_label=True)
    q = to_mat(qubo)
    solutions = solve_gpu(q, offset)

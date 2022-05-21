# QUBOBrute

*GPU Accelerated Solver for Quadratic Binary Optimization Problems.*

-----------------

The goal of this project is to provide a simple, extensible GPU accelerated QUBO Solver for fast debugging of problem definitions.

## Prerequisits

For using the GPU solvers you will need a CUDA capable GPU. This project is tested on an NVIDIA RTX 2060 running on Ubuntu 20.04 LTS. The CPU solvers work on any old Laptop, but of course the faster the better (Looking at you, EPYC 7713...).

## Features

 + Bruteforce solver running in parallel on the CPU.
 + Bruteforce solver running in parallel on an Nvidia GPU.
 + Fast simulated annealing solver running in parallel threads on the CPU.
 + Blazingly fast simulated annealing solver running in parallel threads on the GPU.

## Installation

`conda install --file condarequirements.txt`

In particular, you need

+ `numba`
+ `pyqubo`
+ `cudatoolkit`
+ `scipy`
+ `cupy`

## Usage

    from qubobrute import *
    from pyqubo import Array

    nbits = 30
    x = Array.create("x", shape=(nbits,), vartype="BINARY")
    H = (np.arange(nbits) @ x - 12) ** 2
    model = H.compile()
    qubo, offset = model.to_qubo(index_label=True)
    q = to_mat(qubo)
    energies = solve_gpu(q, offset)
    
Further examples can be found in the `examples` directory.

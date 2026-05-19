# Quantum Tunnelling Boundary-value Solver
This is a repository contains a Python-based numeric solver that computes and visualizes the exact, steady-state wave function ($\psi(x)$) for a particle encountering a 1D rectangular potential barrier.

## Method & Theory
Rather than using time-dependent finite-difference approximations, this script models a stationary state by solving the time-dependent Schrodinger equation adross three piece-wise regions.
By imposing continuity conditions for both $\psi(x)$ and $d\psi/dx$ at the boundaries ($x=a$ and $x=b$), the system is mapped into a $4\times4$ complex liner matrix system and solved using 'numpy.linalg.solve'.

## Requirements
* NumPy
* Matplotlib

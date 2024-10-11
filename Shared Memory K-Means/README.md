# Distributed Memory Heat Equation Solutions

Parallel implementations of various methods for solving the heat equation. The code
included in the _mpi_ directory is designed to run on a computer cluster. In the
context of the course, measurements were taken from 8 8-core nodes.

| Directory | Comment                                                      |
|-----------|--------------------------------------------------------------|
| figures   | Various Figures to accompany the code                        |
| mpi       | Parallel Versions of the heat equation methods               |
| serial    | Serial Versions of the heat equation methods, used as a base |

### Jacobi Method
The parallelization of the jacobi method is fairly straightforward. It requires breaking down the 2D data-grid into
N smaller grids, where N is the number of processes. To achieve this, a 2D cartesian communicator is used.
Each process is responsible for the computations on each subgrid, but there is also a need for communication on each
time iteration. This happens between neighbouring processes across the 4 directions of the cartesian communicator
(e.g. north, west, south, east). Since there are no overlapping data/computation dependencies between processes,
there is no need for non-blocking communication, in contrast to the next cases. The implementation of the above can
is included in _[mpi/Jacobi_mpi.c](mpi/Jacobi_mpi.c)_

### Gauss Seidel SOR Method
For each time iteration of this method, some processes require neighbouring data, which is computed on the same time
iteration. This creates some computation/communication dependencies between processes and requires non-blocking
communication to be implemented. Schematically the above can be summed up in the following figure.

<img src="figures/gsFig.png" width="500">
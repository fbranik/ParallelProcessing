# Distributed Memory Heat Equation Solutions

Parallel implementations of various methods for solving the heat equation. The code 
included in the _mpi_ directory is designed to run on a computer cluster. For the
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
is included in _mpi/Jacobi_mpi.c_

### Gauss Seidel SOR Method
For each time iteration of this method, some processes require neighbouring data, which is computed on the same time
iteration. This creates some computation/communication dependencies between processes and requires non-blocking
communication to be implemented. Schematically the above can be summed up in the following figure.

![Alt text](figures/gsFig.png?raw=true)

Since the communication dependency is in the north and west, the processes in the borders of these directions are the
first to complete the needed computations (since they have no dependencies). This creates a kind of 'wave' of completed
computations on each tome iteration, that starts from the top left. The code for this method using MPI is in 
_mpi/GaussSeidelSOR_mpi.c_

### Red Black SOR Method
This method includes 2 phases of computation and communication per time iteration. Namely, elements of the data grid
are alternatively named red and black. On each time iteration the red elements are updated first. To achieve this, each
process needs some neighbouring black data of the previous time iteration. After the computation of the current red
elements is done, processes need to update their black elements. For this, they need some neighbouring red data that
was just computed. This pattern of communication and computation for each time iteration, can be seen schematically in
the following figure.

![Alt text](figures/rbFig.png?raw=true)

To easily implement this pattern using MPI, custom datatypes where used. Specifically, one of them
includes every other element of a column, while the second one includes every other element of a row. This was done
to easily get either the red or black elements, when communicating with neighbouring processes. This implementation
is included in _mpi/RedBlackSOR_mpi.c_.

### Convergence Test
For all three of the above methods, there is an option to check if the solution has converged at the end of each time
iteration. To achieve this, every process checks if the values of its subgrid have converged. Using MPI_Allreduce, all
processes check whether there is 'global' convergence. If there is, no more time iterations are executed. If not, 
the time-loop is repeated. 

### Measurements
For the context of this exercise, the measured communication/computation/total times were (naively) taken
using reduction to the max value reported by all processes. This presents some contradictions between the sum of
_ComputationTime + CommunicationTime_ and the _TotalTime_. However, there are some insights provided by these
measurements.  

#### Convergence time
In a real use of the above applications, execution with convergence would likely be used. The plot below, shows execution
with convergence for all three methods.

![Alt text](mpi/plots/conv1024AllMethods.png?raw=true)

Parallel computing vs concurrency

Parallel computing, doing more work in less time by leveraging multiple processors / threads, 
while concurrency is about structuring your program to handle multiple tasks efficiently, which can be particularly useful in environments with limited processing resources.

Supercomputer - computer with thousands of cores

CPU vs GPU - less core, many cores
           - optimised hardware, non optimised hardware

Heterogeneous supercomputing - different types of processors

basics of cuda:
initialisation of data
transfer data from cpu to gpu
kernel launch with needed grid size
transfer result back to cpu
reclaim memory from both cpu and gpu
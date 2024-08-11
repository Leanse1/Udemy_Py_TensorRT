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
kernel launch with needed grid/block size
transfer result back to cpu
reclaim memory from both cpu and gpu

CUDA follows SIMT architecute

SIMT - single instruction multiple threads
single instruction is going to run on multiple threads

software -> thread (1024,1024,64), thread block(2^31,65535,65535) , grid
hardware -> cuda core(64), streaming multiprocessor(14), device

CUDA Cores: Parallel processing units in a GPU responsible for executing the threads of a CUDA program simultaneously.
Shared Memory: Fast, on-chip memory shared among threads within the same block, used for efficient data exchange and synchronization.
Registers: High-speed storage in the GPU used to hold temporary variables and intermediate results during computations.
Load/Store Units: Functional units responsible for transferring data between the GPU’s memory and its processing cores.
Warp Schedulers: Components that manage the execution of warps (groups of 32 threads) by scheduling and dispatching them to the CUDA cores.
Special Function Units (SFUs): Specialized hardware units in the GPU designed to perform complex mathematical operations like square roots, sine, and cosine efficiently.

https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Wj6gB_MhhnmGu3OuToAjJg.jpeg

based on CUDA compute capability properties of cuda device is going to vary
gtx 1650 - 7.5
rtx series - 8.0 and above

gtx 1650

major  7
minor .5
totalGlobalMem - which refers to the total amount of global memory available on the GPU, is typically 4 GB (4096 MB).
maxThreadsPerBlock for an NVIDIA GTX 1650 is 1024

maxThreadsDim - x-dimension: 1024, y-dimension: 1024, z-dimension: 64

block boundary value — (1024, 1024, 1024) and the product of all the 3 dim should be less than or equal to 1024.

maxGridsize - x-dimension: 2,147,483,647 (or 2^31 - 1)
              y-dimension: 65,535
              z-dimension: 65,535

Clockrate -  typical clockRate is around 1485 MHz

sharedMemPerBlock - total amount of shared memory available per block on the GPU, measured in bytes. 
For the NVIDIA GTX 1650, the shared memory per block is typically 48 KB (or 49,152 bytes).

The warpSize property indicates the number of threads in a warp on an NVIDIA GPU. 
For the NVIDIA GTX 1650, like most modern NVIDIA GPUs, the warp size is 32 threads.

Streaming multiprocessor: The NVIDIA GTX 1650 features 14 Streaming Multiprocessors (SMs). 
Each SM contains CUDA cores, shared memory, and other resources that work together to execute threads in parallel.

cores: each SM in gtx 1650 contains 64 CUDA cores.

total number of cores = 14*64 = 896 so this device can only run 896 threads in parallel

one block runs in one Streaming multiprocessor; even a block with 1024 threads can only run 64 threads at a time

warps: blocks divided into smaller units called warps, each having 32 consecutive threads
no of warps per block = block size/ warp size = 1024/32 =32
since we have only 64 cores / SM we can only execute 2 warps at a time

for cuda with 7.5
Max Concurrent Threads per Block: 1024 threads  // maximum number of threads that can be launched in a single block.
Max concurrent blocks per SM: 16  
Max Concurrent Warps per SM: 64 warps (2048 threads) // an SM can handle up to 2048 threads concurrently.
Registers per SM: 65,536 registers  // registers are used to store variables and intermediate values during thread execution.
Max Registers per Thread: 255 registers // maximum number of registers that can be allocated to a single thread
Shared Memory per Block: 64 KB (Shared among all threads in the block)  // 64 KB of shared memory per block is shared among all threads in the block. 

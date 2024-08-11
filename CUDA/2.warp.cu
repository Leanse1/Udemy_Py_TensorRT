The instruction architecture of GPU is Single Instruction Multiple Threads (SIMT). 
The threads are executed in a collection called warp. Warp is the basic unit of execution in a GPU. 
Generally, the number of threads in a warp (warp size) is 32. 
Even if one thread is to be processed, a warp of 32 threads is launched by warp scheduler where the 1 thread will be 
the active thread. Hence, we should make sure that all the threads in a warp are active for better utilisation of the 
GPU resources.

Based on the readiness of the warp it is classified in to three:

Selected warp — warp that is actively executing
Eligible warp — warp ready for execution with all its arguments available but awaiting execution
Stalled warp — warp not ready for execution



warps: blocks divided into smaller units called warps, each having 32 consecutive threads
no of warps per block = block size/ warp size = 1024/32 =32
since we have only 64 cores / SM we can only execute 2 warps at a time

Basic unit of execution in SM

gid = blockidx.y * griddim.x * blockdim.x + blockidx.x * blockdim.x * threadidx.x
warpid = threadIdx.x /32   (threadid<2 leads to warp 1)
gbid (blockid) = blockidx.y * griddim.x + blockidx.x

branch efficiency in warp = (number of branches - divergent branches)/number of branches
branch efficiency can be measured using nvprof tool

//compiles my_program.cu into an executable named my_program
nvcc -o my_program my_program.cu

//provide performance metrics and profiling information for the CUDA application.
nvprof ./my_program
nvprof --metrics branch_efficiency my_program

// Eligible to be a warp:
32 cuda cores should free for execution
all arguments for warp should be ready

1 SM -> 64 cores in gtx 1650; can execute 2 warps parallely;


latency: number of clock cycles between instruction being issued and being completed

Compute latency
In Turing architecture, core mathematical operations take 4 clock cycles to execute. 
So, we need 4 warps per warp scheduler in the pipeline to hide this latency.
If there are 4 warp schedulers per SM,
then we need 16 warps or 512 threads (16 x 32) for 100% utilisation of compute cores

In GTX 1650, there are 14 SMs therefore theoretically 224 (14 x 16) warps are needed to hide compute latency.

nvidia-smi -a -q -d CLOCK
get memory value in max clocks



Memory Transfer latency
Base Memory Clock:4 GHz max clock speed for memory transfer
Effective Memory Clock: 128 Gbps 

maximum number of bytes which can be transferred per cycle= 128/4 = 32 bytes/cycle 
now a device with memory latency of 350 clock cycles, 350 * 32 = 11200 bytes can be transferred to SM.
if device is using float which has 4 bytes, we need 11200/4 = 2800 threads to hide memory latency 


#################################################################################################################################

Occupancy: ratio between Active warps per SM and Maximum allowed warps per SM.

Occupancy of our kernel can be calculated as follows,
Calculate the registers and shared memory usage of our kernel with the following command
$nvcc --ptxas-options=-v -o output.exe cuda_file.cu

register per warp = register * 32 (which is warp)
The GTX 1650 has 14 SMs, and each SM has access to 64,000 32-bit registers.
allowed warp based on register= 64000/ register per warp

active warp based on shared memory usage
active blocks = 64000/shared memory 
active warp = active block * 2 (which is warps ber block)

occupancy = active warp/ allowed warp

* one can also use cuda occupancy calculator in google chrome

####################################################################################################################

optimisation with nvprof

modes:
1. summary mode:  high-level overview of your application's performance. 
    It aggregates the data to give a concise report of kernel execution times, memory transfers, and other key metrics.

    nvprof test.out; nvprof --metrics <metric_name> ./your_cuda_application

2. GPU and API trace mode
3. Event metrics summary mode
4. Event metrics trace mode

######################################################################################################################

synchronisation in CUDA

// cudaDeviceSynchronise: Introduce a global synchronise point in host code
// __syncthreads : synchronisation within a block


//REDUCTION ALGORITHM IN CUDA

Reduction is a technique used in parallel computing to combine elements of an array (or any data structure) to produce a single result, such as summing all elements, finding the maximum value, or computing the product of all elements. In CUDA, 
reduction is commonly used for operations like summing values across a large array using the parallel processing power of GPUs.

Parallel Reduction with Synchronization:   Lec 26
Each thread reduces a portion of the data, and results are synchronized using __syncthreads() at each step to ensure correct partial reductions.

Parallel Reduction with Divergence: lec 27
Divergence occurs when threads in the same warp take different execution paths during reduction, leading to inefficient parallelism and underutilization of the GPU.

Parallel Reduction with Loop Unrolling: lec 28
Loop unrolling involves manually unrolling the reduction loop to reduce the number of iterations and synchronization points, leading to improved performance.

Parallel Reduction with Warp Unrolling: lec 29
The reduction process within a warp is unrolled, utilizing warp-level primitives (like __shfl_down_sync) to reduce synchronization overhead and avoid divergence within the warp.

Reduction with Complete Unrolling: lec 30
The entire reduction loop is fully unrolled, eliminating all loops and synchronization steps, allowing for maximum performance at the cost of increased code complexity.

complete unrolling is the best of all parallel reduction alorithm

//DYNAMIC PARALLELISM - lec 32,33
ALLOWS NEW GPU KERNEL TO BE CREATED AND SYNCHRONISED DIRECTLY ON GPU INSTEAD OF CPU
    in dynamic parallelism, grids are divided into two types a) parent grid b) child grid
    parent and child grids share the same global and constant memory but distinct local and shared memory

parent block will be launched from host with one thread block havin 16 threads.
in each grid first thread block will launch child grid which has half the elements in parent grid
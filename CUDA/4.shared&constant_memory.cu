shared memory:
A fixed amount of shared memory is allocated to each block when it starts executing.
this shared memory is shared by all threads in a block
__shared__

can be used to hide performance impact of global memory latency.

bank conflict - lecture 48
Bank conflict: when multiple addresses in a shared memory request fall into the same memory bank, bank conflict occurs, causing request
               to be replayed

how to overcome bank conflict: parallel access, sequential access, broadcast access, extra padding

Types of shared memory

Static Shared Memory
Declaration: The size of the shared memory array is defined at compile time
__shared__ int s[N];

Dynamic Shared Memory
Declaration: The size of the shared memory array is specified at runtime.   
extern __shared__ char s[];

#############################################################################################################################

synchronisation in CUDA

 for ensuring that threads within a block or across blocks operate in a coordinated manner, especially when they share data or 
 need to follow a specific sequence of operations. CUDA provides several synchronization mechanisms to manage the execution of threads, 
 either at the level of a block or a grid.

 __syncthreads();
__syncwarp();
cudaDeviceSynchronize();
cudaStreamSynchronize();

//CONSTANT MEMORY
Constant memory: memory used for data that is read only from device and accessed uniformly by threads in a warp.
Read-Only: Constant memory can only be read by the GPU; it cannot be modified by the GPU once set by the host (CPU).
Small Size: CUDA provides 64 KB of constant memory per device. This limited size makes it suitable for small datasets that are accessed frequently by all threads.
Broadcast Mechanism: When all threads in a warp access the same address in constant memory, the value is broadcast to all threads simultaneously. This makes it very efficient for such access patterns.
Caching: Constant memory is cached, meaning that repeated accesses to the same value by different threads are fast, as the data is stored in a special cache on the GPU.

__constant__ float constArray[256];



//CUDA warp shuffle instructions
CUDA warp shuffle instructions are a set of intrinsics that allow threads within a warp (a group of 32 threads) to exchange data 
directly with each other without using shared memory. These instructions provide a way for threads to communicate more efficiently
by taking advantage of warp-level parallelism, reducing the need for slower shared memory or global memory accesses.

int laneId = threadIdx.x % warpSize;
int value = laneId * 2;
int shuffledValue = __shfl_sync(0xFFFFFFFF, value, 0);
// Now, all threads in the warp will have the value from lane 0.


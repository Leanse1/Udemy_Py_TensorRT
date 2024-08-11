Understanding gld_efficiency, gld_throughput, gld_transactions, and gld_transactions_per_request in CUDA


gld: This typically refers to a Global Load instruction in CUDA, which fetches data from global memory.
Efficiency: A measure of how well a resource is used. In this context, it likely refers to the ratio of useful data loaded to total data loaded.
Throughput: The amount of data processed or tasks completed per unit of time.
Transactions: The number of individual memory accesses.
Transactions per request: The average number of memory accesses per data request.

gld_efficiency
Definition: The ratio of useful data loaded to total data loaded by gld instructions.
Impact: A higher gld_efficiency indicates better memory access patterns and potentially better overall performance. Factors affecting gld_efficiency include:
Coalescing: Efficiently grouping memory accesses together.
Memory access patterns: Strided or random access can impact efficiency.
Cache utilization: Effective use of the L1 and L2 caches.

gld_throughput
Definition: The rate at which data is loaded from global memory using gld instructions.
Impact: A higher gld_throughput implies faster data movement and potentially better performance. Factors affecting gld_throughput include:
Memory bandwidth: The maximum data transfer rate.
Memory latency: The time it takes to access data.
Hardware limitations: The capabilities of the GPU.

gld_transactions
Definition: The total number of gld instructions executed.
Impact: While not a direct performance metric, it can provide insights into memory access patterns and potential bottlenecks.

gld_transactions_per_request
Definition: The average number of gld instructions executed per data request.
Impact: A lower value indicates more efficient memory access patterns, as fewer gld instructions are needed to fetch the required data.
Improving gld Performance
To optimize gld performance, consider the following strategies:
 
Coalescing: Ensure that threads within a warp access consecutive memory locations.
Shared memory: Use shared memory for data that is frequently accessed by multiple threads.
Texture memory: Utilize texture memory for read-only data with spatial locality.
Constant memory: Use constant memory for read-only data that is shared among all threads.


Measuring and Analyzing gld Performance
To measure and analyze gld performance, use profiling tools provided by CUDA, such as NVIDIA Nsight Systems. 
These tools can provide detailed information about memory access patterns, cache utilization, and other relevant metrics.


//TYPES OF MEMORY IN CUDA

SM memory - l1, shared memory, constant, read only
Device memory - l2, global memory, texture memory, constant memory cache 

Fastest - Registers > Cache > main memory > disk memory
Largest - disk > main memory > cache > registers 

Registers: Fastest memory space in GPU; share their lifetime with kernel; hold frequently accessed thread
            1 thread can hold maximum of 255 registers

Local memory: variables which are eligible for registers but cant fit register space.
              not an on-chip memory, so have high latency memory access

shared memory: on chip memory with partition among thread blocks; __shared__
               l1 cache and shared memory for an SM use same on-chip memory

Primary memory:  space accessible to both the host (CPU) and the GPU in CUDA. largest but slowest.
                Data needs to be explicitly copied between the host and device using functions like cudaMemcpy.

zero copy memory: pinned memory; mapped into device address space so that both device and host have direct access
                  host side memory; leverage host memory when there is insufficient device memory; fastest; avoid explicit data transfer; 
                  no need for explicit copy

                  Default pinned memory: Allocated using cudaMallocHost and optimized for host-to-device transfers.
                  Zero-copy pinned memory: Allocated using cudaHostAlloc with the cudaHostAllocMapped flag, 
                                   allowing direct GPU access.

                cudahostallocdefault: same as pinned memory
                cudahotallocportable: pinned memory that can be used by all CUDA contexts
                cudahostallocwritecombined: written by the host and read by the device
                cudahostallocmapped: host memory that is mapped into the device address space
                cudahostetdevicepointer: API function used to obtain the device pointer corresponding to a mapped,
                 pinned host buffer allocated using cudaHostAlloc().

unified memory in cuda: simplifies memory management by providing a single address space accessible to both the CPU and GPU. 
                This means you can allocate memory once and access it from either the host or the device without explicit 
                data transfers. 
                
                cudaMalloc()  


Global memory: 
Aligned memory access : refers to accessing memory at addresses that are multiples of the data type's size. For instance, 
if you are accessing a 4-byte integer, aligned access would occur at addresses that are multiples of 4 (e.g., 0, 4, 8, 12, etc.).

Coalesced memory access : memory access pattern where multiple threads access memory simultaneously.
  Coalescing occurs when multiple memory requests from different threads are combined into a single memory transaction.

L1 cache and L2 cache
Size: L1 Cache is smaller (16KB to 64KB), while L2 Cache is larger (256KB to 2MB or more).
Speed: L1 Cache is faster because it is closer to the CPU core, while L2 Cache is slightly slower but still faster than main memory.
Location: L1 Cache is located within the CPU core, whereas L2 Cache may be on the same chip but outside the core or shared among cores.
Function: L1 Cache serves as the immediate storage for the most critical data and instructions, while L2 Cache holds additional data that is still important but not as frequently accessed.
L1 cache is not involved in global memory writes

uncached memory access:
memory loads that does not utilise l1 cache.

Global Memory: global memory refers to the main memory on the GPU. t's the largest but also the slowest type of memory available to CUDA threads,
                and it's accessible by all threads in a grid.

global memory writes: 
Global memory writes in CUDA are crucial for storing data that needs to persist after a kernel execution.
Coalesced memory access is the key to optimizing these writes, reducing the number of memory transactions, 
and thereby improving performance.
Proper alignment, memory access patterns, and minimizing divergence are all strategies to ensure that global memory writes 
are as efficient as possible.

Array of structure vs structure of array

AoS: An array where each element is a structure containing multiple fields of different data types.
struct Person {
    int age;
    float height;
    char name[20];
};

Person people[10];

SoA: Multiple arrays, each holding a specific field of a record
int ages[10];
float heights[10];
char names[10][20];

Array of Structures (AoS) is often simpler to use and better suited for cases where you need to access and manipulate all fields of an entity together.
Structure of Arrays (SoA) can provide better performance in scenarios where operations involve processing a single field across many entities, particularly in performance-critical applications like GPU programming.
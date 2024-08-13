cuda-stream: sequence of GPU commands that execute in order

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024

__global__ void add(int *a, int *b, int *c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int *h_a, *h_b, *h_c;
  int *d_a, *d_b, *d_c;
  int size = N * sizeof(int);

  // Allocate host memory
  h_a = (int*)malloc(size);
  h_b = (int*)malloc(size);
  h_c = (int*)malloc(size);

  // Initialize host data
  for (int i = 0; i < N; ++i) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  // Allocate device memory
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Create two CUDA streams
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // Copy host data to device using stream1
  cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream1);

  // Launch kernel on stream2
  add<<<N / 256, 256, 0, stream2>>>(d_a, d_b, d_c);

  // Synchronize with stream2
  cudaStreamSynchronize(stream2);

  
  // to overlap data transfers with computation, which can improve the performance of CUDA applications.
  // host pointer should be pinned memory
  cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream1);

  // Synchronize with stream1
  cudaStreamSynchronize(stream1);

  // Check results
  for (int i = 0; i < N; ++i) {
    if (h_c[i] != i * 3) {
      printf("Error at index %d: expected %d, got %d\n", i, i * 3, h_c[i]);
      return 1;
    }
  }

  printf("Success!\n");

  // Free memory and destroy streams
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  return 0;
}

###########################################################################################################################

NULL Stream: default stream that kernel launches and data transfers use if you donot explicitly specify a stream.

Blocking Synchronization:
When using the null stream explicitly (cudaStream_t stream = 0), all operations in the null stream must complete before any operations in other streams can begin. 
Similarly, operations in the null stream will wait for all operations in all other streams to complete before starting.
This behavior ensures that operations in the null stream are synchronized with all other streams, providing a simple and 
safe way to ensure order in your program.

// Launching a kernel in the null stream
myKernel<<<blocks, threads>>>(d_data);
// Memory copy in the null stream
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

cudaStreamNonBlocking flag allows you to create streams that do not synchronize with the null stream or the per-thread default stream,
enabling greater parallelism and concurrency in your CUDA programs.

#include <cuda_runtime.h>
#include <iostream>

__global__ void myKernel(int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] += 1;
}

int main() {
    const int size = 1024;
    int* h_data = new int[size];
    int* d_data;

    cudaMalloc(&d_data, size * sizeof(int));

    // Fill host data
    for (int i = 0; i < size; ++i) {
        h_data[i] = i;
    }

    // Create a non-blocking stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // Asynchronous memory copy in the non-blocking stream
    cudaMemcpyAsync(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Launch a kernel in the non-blocking stream
    myKernel<<<size / 256, 256, 0, stream>>>(d_data);

    // Perform other operations while the kernel and copy are executing

    // Wait for the stream to finish
    cudaStreamSynchronize(stream);

    // Clean up
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    delete[] h_data;

    return 0;
}

##################################################################################################################################

CUDA events are synchronization primitives that allow you to measure the time between different operations on the GPU and 
coordinate the execution of tasks. Events are particularly useful for timing, profiling, and managing dependencies between
asynchronous operations like kernel executions or memory transfers.

cudaEventCreate: Creates an event.
cudaEventDestroy: Destroys an event.
cudaEventRecord: Records an event in a specified stream.
cudaEventSynchronize: Blocks the host until the event is complete.
cudaStreamWaitEvent: Makes a stream wait for an event to complete before continuing.
cudaEventElapsedTime: Computes the time elapsed between two events.

#include <cuda_runtime.h>
#include <iostream>

__global__ void myKernel(int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] += 1;
}

int main() {
    const int size = 1024;
    int* d_data;
    cudaMalloc(&d_data, size * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    myKernel<<<size / 256, 256>>>(d_data);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);

    return 0;
}


ATOMIC FUNCTIONS:

Atomic functions in CUDA are special functions that allow multiple threads to safely read, modify, and write to the same memory
location without causing race conditions. These operations are "atomic" because they are performed as a single, indivisible operation, ensuring that no other thread can access the memory location until the operation is complete.

Why Use Atomic Functions?
In parallel programming, when multiple threads attempt to modify the same memory location simultaneously, 
race conditions can occur, leading to incorrect or inconsistent results. 
Atomic functions prevent these issues by ensuring that each modification is completed before another thread can access 
the same memory location.

Common CUDA Atomic Functions
CUDA provides several atomic functions for different data types and operations:

Integer Atomic Operations:

atomicAdd(int* address, int val): Adds val to the integer value at address and returns the old value.
atomicSub(int* address, int val): Subtracts val from the integer value at address and returns the old value.
atomicExch(int* address, int val): Sets the value at address to val and returns the old value.
atomicMin(int* address, int val): Sets the value at address to the minimum of the current value and val.
atomicMax(int* address, int val): Sets the value at address to the maximum of the current value and val.
atomicAnd(int* address, int val): Performs a bitwise AND between the value at address and val.
atomicOr(int* address, int val): Performs a bitwise OR between the value at address and val.
atomicXor(int* address, int val): Performs a bitwise XOR between the value at address and val.
atomicCAS(int* address, int compare, int val): Compares the value at address with compare, and if they are equal, sets the value at address to val. It returns the old value.
Floating-Point Atomic Operations:

atomicAdd(float* address, float val): Adds val to the floating-point value at address and returns the old value. (Supported on devices of compute capability 2.0 and higher.)
Double-Precision Atomic Operations:

atomicAdd(double* address, double val): Adds val to the double-precision value at address and returns the old value. (Supported on devices of compute capability 6.0 and higher.)
Unsigned Long Long Atomic Operations:

atomicAdd(unsigned long long int* address, unsigned long long int val): Adds val to the value at address and returns the old value.
atomicExch(unsigned long long int* address, unsigned long long int val): Sets the value at address to val and returns the old value.
Example Usage of Atomic Functions
Here’s an example of using atomicAdd in a CUDA kernel:

#include <cuda_runtime.h>
#include <iostream>

__global__ void atomicAddKernel(int* d_data) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Perform atomic addition
    atomicAdd(&d_data[0], 1);
}

int main() {
    const int arraySize = 1;
    int h_data[arraySize] = {0};
    int* d_data;

    cudaMalloc(&d_data, arraySize * sizeof(int));
    cudaMemcpy(d_data, h_data, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int numBlocks = 1;

    // Launch the kernel
    atomicAddKernel<<<numBlocks, blockSize>>>(d_data);

    // Copy result back to host
    cudaMemcpy(h_data, d_data, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Result after atomicAdd: " << h_data[0] << std::endl;

    // Clean up
    cudaFree(d_data);

    return 0;
}




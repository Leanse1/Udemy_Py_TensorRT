// visual studio >> file >> new >> project >> template bar >> nvidia >> cuda >> myfirstproj (kernel.cu instead of cpp)

#include <iostream>
#include <cuda_runtime.h>

// __global__ keyword indicates that this function will be executed on the device (GPU).
__global__ void helloWorld() {
    printf("Hello, World from the GPU!\n");
}

//The host code runs on the CPU and is responsible for launching the kernel.
int main() {
    // Launch kernel on the device
    // <<<number of blocks, number of threads per block
    helloWorld<<<1, 1>>>();

    // Wait for the GPU to finish before accessing the output
    cudaDeviceSynchronize();

    cudaDeviceReset();

    return 0;
}


// Threads: Threads are the individual execution units within a block. 
// Each thread has a unique index within its block, which can be accessed using threadIdx.x, threadIdx.y, and threadIdx.z.

// Block: A block is a group of threads that execute the kernel function. Each block can also be one-dimensional, two-dimensional, or three-dimensional.
// Block Dimensions: The dimensions of a block are defined by dim3 blockDim and can be accessed within a kernel using blockDim.x, blockDim.y, and blockDim.z.
// Block Index: Each block within a grid has a unique index that can be accessed using blockIdx.x, blockIdx.y, and blockIdx.z.

// Grid: A grid is a collection of blocks. It represents the entire set of threads that will execute a given kernel function.
// A grid can be one-dimensional, two-dimensional, or three-dimensional, depending on the problem's requirements and the 
// kernel's configuration.
// Grid Dimensions: The dimensions of a grid are defined by dim3 gridDim and can be accessed within a kernel 
// using gridDim.x, gridDim.y, and gridDim.z.


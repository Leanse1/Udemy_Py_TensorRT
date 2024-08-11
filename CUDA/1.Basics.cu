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
// helloWorld<<<1, 20>>>(); output helloworld will be repeated 20 times

//int nx, ny
//nx =16   //threads in x 
//ny = 4   // threads in y
//dim3 block(8,2)  // (8,2,1)
//dim3 grid(nx/block.x,ny/block.y)   // important formula
// helloWorld<<<grid, block>>>();  //hellocuda will be repeated 64 times which mens 64 threads

//dim3 block(4)   (4,1,1)
//dim3 grid(8)    (8,1,1)
// helloWorld<<<grid, block>>>();


// Grid: A grid is a collection of blocks. It represents the entire set of threads that will execute a given kernel function.
// A grid can be one-dimensional, two-dimensional, or three-dimensional, depending on the problem's requirements and the 
// kernel's configuration.
// Grid Dimensions: The dimensions of a grid are defined by "dim3 gridDim" and can be accessed within a kernel 
// using gridDim.x, gridDim.y, and gridDim.z.

// https://anuradha-15.medium.com/cuda-thread-indexing-fb9910cba084

// gridx = number of threads in x dimension/ number of block in x dimension 

// maximum threads per block can be only upto 1024 for x and y dimension, for z dimension upto 64 thread
// max number of threads per block along 3 dim combined should be less than 1024 (x*y*z = 1024)

// 1D Grid: Up to 2^31 blocks
// 2D Grid: Up to 65535 blocks in each dimension.
// 3D Grid: Up to 65535 blocks in each dimension.

//threadx,blockidx, griddimx namings - lec 6, 7

//threadidx: Refers to the thread ID with in a block. ranges from 0 to 31 in each block.
// blockIdx.x gives the block index in the x-dimension. Refers to the block ID in a grid and it starts from 0.
// blockDim.x Refers to the maximum number of threads in a block in all the dimension and it starts from 1.
//            gives the number of threads in the x-dimension of the block.
//            same for all threads in a grid
// blockDim.y gives the number of threads in the y-dimension of the block and it starts from 1.
//             same for all threads in a grid
// griddim - number of thread blocks in each grid dimension


global index calculation formula for 1d grid:
https://slideplayer.com/slide/5225331/16/images/8/4+blocks%2C+each+having+8+threads.jpg

global index calculation formula for 2d grid with 1D block :
index = gridDimx * blockDimx * blockId.y + blockIdx * blockDim.x + threadIdx

global index calculation formula for 2d grid with 2D block :
https://kdm.icm.edu.pl/Tutorials/GPU-intro/GPU_images/CUDA_Thread_Block_Idx.png
totalindex = blockDim.x * threadIdx.y + threadIdx.x
gid = totalindex + x + y (from above link)

Guide line for grid and block size

1. keep the number of threads per block a multiple of warp size (32)
2. start with atleast 128 threads per block

#####################################################################################################################

cudaMalloc: Used to allocate memory on the GPU.
cudamalloc(typecastpointer, size)
cudaMemset(void *devPtr, int value, size_t count); // point to gpu, dtype, size

cudaFree: cudaFree is used to free GPU memory that was previously allocated using cudaMalloc
cudaFree(void *devPtr);


cudamemset: set GPU memory space to a specific value

cudaMemcpy: Used to transfer data between the host and GPU or between different GPU memory regions.
cudaMemcpy(destination point, source point, size in byte, direction)
direction - hosttodevice, devicetohost, devicetodevice

#include <iostream>
#include <cuda_runtime.h>

__global__ void addOne(int *d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_array[idx] += 1;
    }
}

int main() {
    int n = 100;
    int h_array[100];  // Host array
    int *d_array;      // Device array

    // Initialize the host array with some values
    for (int i = 0; i < n; ++i) {
        h_array[i] = i;
    }

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_array, n * sizeof(int));  // allocating memory to a mmory

    // Copy data from the host to the device
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel to add 1 to each element
    addOne<<<blocksPerGrid, threadsPerBlock>>>(d_array, n);

    // Copy the result back from the device to the host
    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result after adding 1 to each element:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // Free the memory allocated on the GPU
    cudaFree(d_array);

    return 0;
}


CUDA validity check: lecture 14
CUDA error handling: lec 15
CUDA timing : lec 16   //clock_t 
run program with multiple block, grid thrad to choose best configuration based on execution times


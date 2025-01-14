#include <iostream>
#include <math.h>
#include <chrono>
// Kernel function to add the elements of two arrays
//Exercise 3 from: https://developer.nvidia.com/blog/even-easier-introduction-cuda/

/*
Print the value of threadIdx.y or threadIdx.z (or blockIdx.y) in the kernel. (Likewise for blockDim and gridDim). 
Why do these exist? How do you get them to take on values other than 0 (1 for the dims)?
*/

/*
Answer:
They exist for kernels of different dimensions, we'd have them take on values other than 0/1 when we ultimately change the inputs
to the kernel on the function call.

ThreadIdx.x prints the actual thread index, .y and .z will print zero because they're aren't any threads along those dimensions 
    because we're only passing a number of blocks == 1 we've got no threads along any other dimension
BlockDim.x = 256 and .y&.z=1 because the blocksize is 256 in a single dimension, we could make .y and .z take on !1 via defining those dimensions
gridDim.x = 4096 and .y & .z=1 because once again we define the grid just along a single dimension (just like we did with the blocks

ThreadId/BlockId exist for multi-dimensional kernels and are used to index threads and blocks in 2D or 3D grids.
Defaulting to 0/1 respectively when we don't define those additional dimensions

1D: Token (word) Processing, 2D: Image processing, 3D: Volumetric/Physics Sims

Example of 3D definitions:
        dim3 blockDim(16, 16, 1); // Block of 16x16 threads
        dim3 gridDim(32, 32, 1);  // Grid of 32x32 blocks
        kernel<<<gridDim, blockDim>>>(...);

Ex:
##add_v5_v1
threadIdx.x: 126 || threadIdx.y: 0 || threadIdx.z: 0
threadIdx.x: 127 || threadIdx.y: 0 || threadIdx.z: 0
threadIdx.x: 64 || threadIdx.y: 0 || threadIdx.z: 0
threadIdx.x: 65 || threadIdx.y: 0 || threadIdx.z: 0

##add_v5_v2
blockDim.x: 256 || blockDim.y: 1 || blockDim.z: 1
blockDim.x: 256 || blockDim.y: 1 || blockDim.z: 1

##add_v5_v3
gridDim.x: 4096 || gridDim.y: 1 || gridDim.z: 1
gridDim.x: 4096 || gridDim.y: 1 || gridDim.z: 1


*/

__global__
void add(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    //printf("threadIdx.x: %d || threadIdx.y: %d || threadIdx.z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    //printf("blockDim.x: %d || blockDim.y: %d || blockDim.z: %d\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim.x: %d || gridDim.y: %d || gridDim.z: %d\n", gridDim.x, gridDim.y, gridDim.z);
    
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20;
    float* x, * y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel on 1M elements on the GPU
    /*
    Together, the blocks of parallel threads make up what is known as the grid.
    Since I have N elements to process, and 256 threads per block, I just need to calculate the number of blocks to get at least N threads.
    I simply divide N by the block size (being careful to round up in case N is not a multiple of blockSize).
    */
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add << <numBlocks, blockSize >> > (N, x, y);
    /*
    This is called the execution configuration, and it tells the CUDA runtime how many parallel threads to use for the launch on the GPU.
    There are two parameters here, but let’s start by changing the second one: the number of threads in a thread block.
    CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, so 256 threads is a reasonable size to choose.
    */

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    /*
    Just one more thing: I need the CPU to wait until the kernel is done before it accesses the results
    (because CUDA kernel launches don’t block the calling CPU thread).
    To do this I just call cudaDeviceSynchronize() before doing the final error checking on the CPU.
    */

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Kernel execution time: " << elapsed.count() << " seconds\n";
    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
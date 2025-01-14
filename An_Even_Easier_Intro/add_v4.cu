#include <iostream>
#include <math.h>
#include <chrono>
// Kernel function to add the elements of two arrays
//Exercise 2 from: https://developer.nvidia.com/blog/even-easier-introduction-cuda/

/*
Experiment with printf() inside the kernel. 
Try printing out the values of threadIdx.x and blockIdx.x for some or all of the threads. 
Do they print in sequential order? Why or why not?
*/

/*
Answer:
I believe they should be printing out of order because the GPU is executing the kernel 
with all threads in parallel so some threads will finish before others do thus they won't print sequentially.
1. CUDA kernels execute on a grid of thread blocks, where each block contains multiple threads.
2. Threads within a block execute in warps (groups of 32 threads). Each warp executes instructions in lockstep, 
    but different warps within the same block or across different blocks may execute asynchronously.
3. The CUDA runtime does not enforce a strict order of execution for threads across blocks or even warps within a block. 
    This means thread execution is interleaved or parallelized in a way that maximizes hardware utilization.
4. Within a warp (32 threads), execution is synchronized and deterministic, 
    but warps within the same block or across different blocks may execute out of order.
Ex:
blockIdx.x: 125, threadIdx.x: 253
blockIdx.x: 125, threadIdx.x: 254
blockIdx.x: 125, threadIdx.x: 255
blockIdx.x: 124, threadIdx.x: 32
blockIdx.x: 124, threadIdx.x: 33
blockIdx.x: 124, threadIdx.x: 34
blockIdx.x: 124, threadIdx.x: 35

*/

__global__
void add(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //printf("Index: %d\n", index);
    //printf("Stride: %d\n", stride);
    //printf("blockIdx.x: %d\n", blockIdx.x);
    //printf("threadIdx.x: %d\n", threadIdx.x);
    //printf("blockDim.x: %d\n", blockDim.x);
    //printf("gridDim.x: %d\n", gridDim.x);
    printf("blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x, threadIdx.x);
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
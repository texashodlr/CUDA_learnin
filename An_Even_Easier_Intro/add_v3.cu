#include <iostream>
#include <math.h>
#include <chrono>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float* x, float* y)
{
    //New FOR Loop (Grid-Stride Loop)
    //threadIdx.x contains the index of the current thread within its block
    //1 Thread block with 256 threads! Multiples of 32 also
    //the idea is that each thread gets its index by computing the offset to the beginning of its block 
    // (the block index times the block size: blockIdx.x * blockDim.x) and adding the thread�s index within the block (threadIdx.x).
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //blockDim.x contains the number of threads in the block
    //CUDA provides gridDim.x, which contains the number of blocks in the grid, 
    // and blockIdx.x, which contains the index of the current thread block in the grid.
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];

    /* Old FOR loop
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
    */
}

int main(void)
{
    int N = 1 << 20;
    float* x, * y;

    // Allocate Unified Memory � accessible from CPU or GPU
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
    There are two parameters here, but let�s start by changing the second one: the number of threads in a thread block.
    CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, so 256 threads is a reasonable size to choose.
    */

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    /*
    Just one more thing: I need the CPU to wait until the kernel is done before it accesses the results
    (because CUDA kernel launches don�t block the calling CPU thread).
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
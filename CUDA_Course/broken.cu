#include <iostream>
#include <math.h>

//This isn't broken, fyi//

// Kernel definition
__global__ void addArrays(float* a, float* b, float* c, int size) {
    // Calculate unique index for this thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't go past array bounds
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main(void) {
    int N = 1 << 20; // Bitwise shift left op. 1,048,576
    float* d_a, * d_b, * d_c;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&d_a, N * sizeof(float)); //2^20 * 4B ==4Mib
    cudaMallocManaged(&d_b, N * sizeof(float));
    cudaMallocManaged(&d_c, N * sizeof(float));
    //net is 12MiB

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        d_a[i] = 1.0f;
        d_b[i] = 2.0f;
        d_c[i] = 3.0f;
    }

    // In main function:
    dim3 blockSize(256);  // 256 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);  // Calculate needed blocks
    addArrays << <gridSize, blockSize >> > (d_a, d_b, d_c, N);

    cudaDeviceSynchronize();
    printf("Kernel completed!");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
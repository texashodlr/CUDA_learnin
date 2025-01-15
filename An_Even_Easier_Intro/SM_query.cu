#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Number of Streaming Multiprocessors (SMs): " << prop.multiProcessorCount << "\n";

    return 0;
}

//36 SMs on my GeForce 4070
//128 Cuda Cores/SM so 128*36=4608 CUDA Cores in this laptop

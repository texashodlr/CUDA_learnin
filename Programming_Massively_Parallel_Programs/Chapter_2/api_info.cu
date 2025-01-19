//This file is just some notes on the CUDA API from Chapter 2

//Figure 2.5: Alternative Method:
//Loop Parallelism
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

__host__
void vecAdd(float* A_h, float* B_h, float* C_h, int n){
//This function is HOST CODE (Stub!)

int size = n* sizeof(float);
float *A_d, *B_d, *C_d;

//Memory Allocations
cudaMalloc((void**) &A_d, size);
cudaMalloc((void**) &B_d, size);
cudaMalloc((void**) &C_d, size);

cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

//Kernel Invocation
vecAddKernel << <ceil(n / 256.0), 256 >> > (A_d, B_d, C_d, n);

cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

cudaFree(A_d);
cudaFree(B_d);
cudaFree(C_d);

}

int main(void) {

	int N = 1 << 20;
	
	float* A = new float[N];
	float* B = new float[N];
	float* C = new float[N];

	// Initialize A and B with some values
	for (int i = 0; i < N; i++) {
		A[i] = 1.0f;
		B[i] = 2.0f;
	}

	vecAdd(A, B, C, N);

	// Free host memory
	delete[] A;
	delete[] B;
	delete[] C;

	return 0;

}


	//cudaMalloc();
	/*
	1. Allocates objects in the device global memory
	2. Two params:
		1. Addr of a pointer to the allocated object (casting to (void **)
		2. Size of allocated object in terms of bytes
	The cudaMalloc function wites to the pointer variable whose addr is given as the first param,
		this allows the return value to report errors
	*/


	//cudaFree();
	/*
	Frees objects from device global memory
		Pointer to freed object
	*/

	//Explanation:
	/*
	First arg passed to cudaMal is the addr of the pointer A_d (&A_d) casted to a void pointer.
		When it returns A_d will point to the device global memory region allocated to the A vector.
	Second arg passed is the size of the region to be allocated (4-bytes)

	CudaFree doesn't need to change the value of A_d, just use the value of A_d to return the allocated memory back to the available pool
		So only the value, not the address is passed.
	*/


	//cudaMemcpy();
	/*
	1. Memory data transfer
	2. Requires four params:
		1. pointer to dest
		2. pointer to source
		3. Number of bytes copied
		4. Type/Direction of transfer
	*/

	//Error handling
	/*
	
	cudaError_t err = cudaMalloc((void**) &A_d,size);
	if (error!=cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
		exit(EXIT_FAILURE);
	}
	*/
	

	//blockDim variable struct with three unsigned int fields (x,y,z)
	//threadIdx.x/y/z [0...255]
	//blockIdx.x/y/z -- gives all threads in a block a common block coord [0..N-blocks]
	// Could calc a unique global index: i = blockIdx.x * blockDim + threadIdx.x
	// Thread seven block 1 == i = 1*256+7 = 263
	//
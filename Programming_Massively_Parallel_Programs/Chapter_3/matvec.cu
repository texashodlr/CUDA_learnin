#include <chrono>
#include <iostream>

//This Code is for Exercise 2
__global__
void MatVecMulKernel(float* OUT, float* IN_M, float* IN_V, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < N) {
		float Pvalue = 0;
		for (int k = 0; k < N; ++k) {
			//printf("IN_M: %f\n", IN_M[row * N + k]);
			//printf("IN_V: %f\n", IN_V[k]);
			Pvalue += IN_M[row * N + k] * IN_V[k];

		}
		OUT[row] = Pvalue;
	}
}

int main(void) {

	//Using the chapter 3 example matrix
	const int LENGTH = 3;
	const int VECTOR_SIZE = LENGTH; // 2x1
	const int MATRIX_SIZE = VECTOR_SIZE * VECTOR_SIZE; // 2x2
	size_t vector_bytes = VECTOR_SIZE * sizeof(float);
	size_t matrix_bytes = MATRIX_SIZE * sizeof(float); //saves space during cudaMalloc

	//Allocating memory on the host for the matrices: B, C, A	
	float h_B[MATRIX_SIZE], h_C[VECTOR_SIZE], h_A[VECTOR_SIZE];

	//Filling the matrices with values
	for (int i = 0; i < MATRIX_SIZE; i++) {
		h_B[i] = static_cast<float>(i + 1);
	}
	//Filling the Vectors with values
	for (int i = 0; i < VECTOR_SIZE; i++) {
		h_C[i] = static_cast<float>(i + 1);
		h_A[i] = 0.0f;
	}
	
	//Now allocating memory for matrices on the device
	float* d_B, * d_C, * d_A;
	cudaMalloc((void**)&d_B, matrix_bytes);
	cudaMalloc((void**)&d_C, vector_bytes);
	cudaMalloc((void**)&d_A, vector_bytes);

	//Now copying the input matrices M and N from H to D
	cudaMemcpy(d_B, h_B, matrix_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, vector_bytes, cudaMemcpyHostToDevice);

	//Now we're sizing the blocks and grids
	dim3 blockDim(1, LENGTH);
	dim3 gridDim(1, (LENGTH + blockDim.y - 1) / blockDim.y);

	//Clock Start
	auto start = std::chrono::high_resolution_clock::now();

	//Function Call
	MatVecMulKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, LENGTH);

	//Clock End
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Function execution time: " << elapsed.count() << " seconds\n";

	//Copying the device's P to host P
	cudaMemcpy(h_A, d_A, vector_bytes, cudaMemcpyDeviceToHost);

	//Now print all matrix values, remembering: Pvalue += M[row * Width + k] * N[k * Width + col];
	//Starting with M
	std::cout << "Matrix B:\n";
	for (int r = 0; r < LENGTH; r++) {
		for (int k = 0; k < LENGTH; k++) {
			std::cout << h_B[r * LENGTH + k] << " ";
		}
		std::cout << "\n";
	}

	//Then N
	std::cout << "Vector C:\n";
	for (int c = 0; c < LENGTH; c++) {
		std::cout << h_C[c] << " ";
		std::cout << "\n";
	}

	//..and finally P
	std::cout << "Vector A:\n";
	for (int c = 0; c < LENGTH; c++) {
		std::cout << h_A[c] << " ";
		std::cout << "\n";
	}


	//Freeing the device memory
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_A);

	return 0;

}
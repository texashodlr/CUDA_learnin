#include <chrono>
#include <iostream>

//Exericse 1.B: Write a kernel that has each thread produce one output matrix col:
__global__
void MatrixMulKernelCol(float* M, float* N, float* P, int Width) {
	int	col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < Width) {
		for (int row = 0; row < Width; ++row) {
			//Would compute P[row][col]
			float Pvalue = 0;
			for (int k = 0; k < Width; ++k) {

				Pvalue += M[row * Width + k] * N[k * Width + col];
			}
			P[row * Width + col] = Pvalue;
		}
	}
}

int main(void) {

	//Using the chapter 3 example matrix
	const int BLOCK_WIDTH = 2;
	const int WIDTH = BLOCK_WIDTH * BLOCK_WIDTH;
	const int MATRIX_SIZE = WIDTH * WIDTH;
	size_t bytes = MATRIX_SIZE * sizeof(float); //saves space during cudaMalloc

	//Allocating memory on the host for the matrices: M, N, P	
	float h_M[MATRIX_SIZE], h_N[MATRIX_SIZE], h_P[MATRIX_SIZE];

	//Filling the matrices with values
	for (int i = 0; i < MATRIX_SIZE; i++) {
		h_M[i] = static_cast<float>(i + 1);
		h_N[i] = static_cast<float>(i + 1);
		h_P[i] = 0.0f;
	}
	//Take note that this for-loop formats the matrices as [ 1 2 3 4 / 5 6 7 8 / ... ] that is M=N

	//Now allocating memory for matrices on the device
	float* d_M, * d_N, * d_P;
	cudaMalloc((void**)&d_M, bytes);
	cudaMalloc((void**)&d_N, bytes);
	cudaMalloc((void**)&d_P, bytes);

	//Now copying the input matrices M and N from H to D
	cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

	//Now we're sizing the blocks and grids
	//Modified for X!
	dim3 blockDim(16);
	dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x);

	//Clock Start
	auto start = std::chrono::high_resolution_clock::now();

	//Function Call
	MatrixMulKernelCol << <gridDim, blockDim >> > (d_M, d_N, d_P, WIDTH);

	//Clock End
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Function execution time: " << elapsed.count() << " seconds\n";

	//Copying the device's P to host P
	cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);

	//Now print all matrix values, remembering: Pvalue += M[row * Width + k] * N[k * Width + col];
	//Starting with M
	std::cout << "Matrix M:\n";
	for (int r = 0; r < WIDTH; r++) {
		for (int k = 0; k < WIDTH; k++) {
			std::cout << h_M[r * WIDTH + k] << " ";
		}
		std::cout << "\n";
	}

	//Then N
	std::cout << "Matrix N:\n";
	for (int c = 0; c < WIDTH; c++) {
		for (int k = 0; k < WIDTH; k++) {
			std::cout << h_N[c * WIDTH + k] << " ";
		}
		std::cout << "\n";
	}

	//..and finally P
	std::cout << "Matrix P:\n";
	for (int r = 0; r < WIDTH; r++) {
		for (int c = 0; c < WIDTH; c++) {
			std::cout << h_P[r * WIDTH + c] << " ";
		}
		std::cout << "\n";
	}


	//Freeing the device memory
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);

	return 0;

}
//Output
/*

Function execution time: 2.8087e-05 seconds
Matrix M:
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
Matrix N:
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
Matrix P:
90 100 110 120
202 228 254 280
314 356 398 440
426 484 542 600

*/
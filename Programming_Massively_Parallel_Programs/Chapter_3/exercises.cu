//In the chapter we produced a matMul kernel which has each thread produce one output matrix element:
__global__
void MatrixMulKernel(float* M, float* N, float* P, int Width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int	col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < Width) && (col < width)) {
		float Pvalue = 0;
		for (int k = 0; k < Width; ++k) {
			//Beginning Element of row 1 is M[1*Width] accessing the kth element of mth rowth row is M[row*Width+k]
			//Beginning element of colth column is the colth element of row 0 which is N[col]
				//Kth element of the colth column is N[k*Width+col] which means we skip over whole rows
			Pvalue += M[row * Width + k] * N[k * Width + col];
		}
		P[row * Width + col] = Pvalue;
	}
}

//Exericse 1.A: Write a kernel that has each thread produce one output matrix row:
__global__
void MatrixMulKernelRow(float* M, float* N, float* P, int Width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (row < Width) {
		for (int col = 0; col < Width; ++col) {
			//Would compute P[row][col]
			float Pvalue = 0;
			for (int k = 0; k < Width; ++k) {

				Pvalue += M[row * Width + k] * N[k * Width + col];
			}
			P[row * Width + col] = Pvalue;
		}	
	}
}
//Launch Configs:
// Each thread handles a single row
int blockSize = 4;  // Number of threads per block, originally 4 th/bl with 4 blocks now still 4 threads per block but just a single block
int numBlocks = (Width + blockSize - 1) / blockSize;  // Total blocks needed, technically 1

dim3 blockDim(blockSize);  // Threads in y-dimension
dim3 gridDim(numBlocks);   // Blocks in y-dimension

MatrixMulKernelRowWise << <gridDim, blockDim >> > (M, N, P, Width);

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
//Launch Configs:
// Each thread handles a single column
int blockSize = 4;  // Number of threads per block, originally 4 th/bl now still 4 threads per block
int numBlocks = (Width + blockSize - 1) / blockSize;  // Total blocks needed, technically 1

dim3 blockDim(blockSize);  // Threads in x-dimension
dim3 gridDim(numBlocks);   // Blocks in x-dimension

MatrixMulKernelColWise << <gridDim, blockDim >> > (M, N, P, Width);

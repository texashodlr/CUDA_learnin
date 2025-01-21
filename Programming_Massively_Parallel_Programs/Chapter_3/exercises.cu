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


//Exericse 1.C: What are the pro/cons of Col-wise and Row-wise?
/*
--Row_wise--
P:
	More memory efficient as threads in a warp/group would be accessing consecutive memory locations (speed) -- coalesced memory
		Row access for M but not for N so memory inefficient for that matrix (Stride access and thus non-coalesced)
	Row-wise is generally how most things are laid out (conceptually easier for me at least haha)
C:
	Non-coalesced for N, and divergence of work efforts depending on how elements are laid out in a matrix, some threads may do more work than others!
--Col_wise--
P:
	Memory eff for N (col-wise) but not for M (row-wise)
	Wide v tall matrices
C:
	Uneven work/thread divergence (like Row_wise).


*/

//Exercise 2:
/*

A matmul takes an input mat B and vector C and produces one output vector A,
Each element of the output vector A is the dot product of one row of the input mat B and C that is A[i]= EjB[i][j] + C[j]
We'll only handle square matrices 
Write a kernel with four params: point to the out mat, pointer to the in mat, point to the in vec and number of elements in each dim
	Use one thread to calculate an output vector element

*/
void MatVecMulKernel(float* OUT, float* IN_M, float* IN_V, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (row < N){
		float Pvalue = 0;
		for (int k = 0; k < N; ++k) {
			Pvalue += IN_M[row * N + k] * IN_V[k];
		}
	OUT[row] = Pvalue;
	}
}

//Exercise 3:
/*

Consider this cuda kernel

__global__
void foo_kernel(float* a, float* b, unsigned int M, unsigned int N){
	unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;
	if( row < M && col < N){
		b[row*N + col] = a[row*N+col]/2.1f + 4.8f;
	}
}

void foo(float* a_d, float* b_d){
	unsigned int M = 150;
	unsigned int N = 300;
	dim3 bd(16, 32);
	dim3 gd((N-1)/16 + 1, (M-1)/32 + 1);
	foo_kernel<<<gd, bd>>>(a_d, b_d, M, N);
}

\\\Questions///
1. Threads per Block?
	.X = 16 & .y = 32 so 16*32 = 512 threads per block
2. Number of threads in the grid?
	grid.x = (300-1)/16 + 1 == 19 -- 20
	grid.y = (150-1)/32 + 1 == 5 -- 6
	19 * 5 = 95 Blocks * 512 = 48640
	so 20*6 = 120 Blocks, 120*512 = 61,440 Threads in the grid
3. Number of blocks in the grid?
	120
4. Number of threads that execute code on line 5?
	Line 5: b[row*N + col] = a[row*N+col]/2.1f + 4.8f;
	..Well.. Line 5 is gatekept by: if( row < M && col < N) and M == 150, N=300 and 150*300 = 45K!

*/

//Exercise 4:
/*

W = 400, H = 500, (200K elements) specifiy the array index of the element at row 20 and col 10.
A. In row-major order: M[20*400+10] == 8010th element
B. In Col-major order: M[20+10*500] == 5020th element

*/

//Exercise 5:
/*

3D Tensor now, W=400, H=500, D=300, laying it out as a 1D array in row-major order whats the index of X=10 y=20, Z=5? (Row 20, Col 10, Depth 5
60M elements

P[plane*m*n+row*m+col]

P[5*400*500+20*400+10] = 1,008,010th element
*/
//General Chapter 3 Notes

/*

dim3 dimGrid(32, 1, 1); --1D grid of 32 Blocks
dim3 dimBlock(128, 1, 1); 1D Block of 128 threads
32 x 128 == 4096
dim3 is the type, dimGrid/Block are arbitrary

Can also use mathematical funcs
dim3 dimGrid(ceil(n/256.0), 1, 1);
dim3 dimBlock(256, 1, 1);

gridDim.x limits are 1 to 2^31 - 1, .y and .z are 2^16

Blocks are limited to 1024 threads

(32, 32, 2) isn't allowed as it == 2048

Grid and Block dimensionality need not be equal.

Grid(2, 2, 1); and Block(4, 2, 2); are legal

Thread with label (1,0,2) would be threadIdx.x=1, .y=0 .z=2

Example of a 2D Picture:
	1: vertical row coord = blockIdx.y*blockDim.y+threadIdx.y
	2: horizonal col coor = blockidx.x*blockDim.x+threadIdx.x
dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);
dim3 dimBlock(16, 16, 1);
colorToGrayScaleConversion<<dimGrid,dimBlock>>>(Pin_d, Pout_d, m, n);

to process a 1500x2000 (3M pixel) photo we would generate 11,750 blocks, 94-y, 125-x (grid) and 16-y, 16-x (block) == 3.008M

Now we need to linearize/flatten the 2D photo(array) into a 1D. 
	Row-major discussion, Row*Width+Col, eg: 4x4 Array want 9th element: M9 = 2*4+1 (M2,1)
	

3D -- Plane
int plane = blockIdx.z*blockDim.z + threadIdx.z
linearized access would be (for 3D Array P) P[plane*m*n+row*m+col]

Matrix Multiplication:
1. We can use each thread to calculate one P element.
2. Row/Col indices are thusly:
	row = blockIdx.y*blockDim.y+threadIdx.y
	col = blockIdx.x*blockDim.x+threadIdx.x

*/

//Input image is encoded as unsigned chars [0, 255]
//Each pixel is 3 consecutive chars for the 3 channels (RGB).
//Each pixel is 1 byte
__global__
void colortoGrayscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, in height) {
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	if (col < width && row < height) {
		//Get 1D offset for the grayscale image
		int grayOffset = row * width + col;
		//One can think of RGB image having a CHANNELS
		//times more columns than the grayscale image
		int rgbOffset = grayOffset * CHANNELS;
		unsigned char r = Pin[rgbOffset];
		unsigned char g = Pin[rgbOffset + 1];
		unsigned char b = Pin[rgbOffset + 2];
		//Perform the rescaling and store it
		//We multiply by floating point constants
		Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
	}
}

__global__
void blurKernel(unsigned char* in, unsigned char* out, int w, int h) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < w && row < h) {
		int pixVal = 0;
		int pixels = 0;

		//Get average of the surrounding BLUR_SIZE x BLUR_SIZE Box
		for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
			for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
				int curRow = row + blurRow;
				int curCol = col + blurCol;
					//Verify we have a valid image pixel
				if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
					pixVal += in[curRow * w + curCol];
					++pixels; //Keeps track of the number of pixels in the avg
				}
			}
		}
		//Write our new pixel value out
		out[row * w + col] = (unsigned char)(pixVal / pixels);
	}
}

__global__
void MatrixMulKernel(float* M, float* N, float* P, int Width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int	col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < Width) && (col < width)) {
		float Pvalue = 0;
		for (int k = 0; k < Width; ++k;) {
			//Beginning Element of row 1 is M[1*Width] accessing the kth element of mth rowth row is M[row*Width+k]
			//Beginning element of colth column is the colth element of row 0 which is N[col]
				//Kth element of the colth column is N[k*Width+col] which means we skip over whole rows
			Pvalue += M[row * Width + k] * N[k * Width + col];
		}
		P[row * Width + col] = Pvalue;
	}
	

}
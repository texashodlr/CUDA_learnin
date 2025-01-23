//Exercise 1: Consider the following Kernel and Host Code
__global__ void foo_kernel(int* a, int* b) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x < 40 || threadIdx.x >= 104) {
		b[i] = a[i] + 1; //Line 4
	}
	if (i % 2 == 0) {
		a[i] = b[i] * 2; //Line 7
	}
	for (unsigned int j = 0; j < 5 - (i % 3); ++j) {
		b[i] += j;
	}
}

void foo(int* a_d, int* b_d) {
	unsigned int N = 1024;
	foo_kernel << <(N + 128 - 1) / 128, 128 >> > (a_d, b_d);
}

/*

Exercise 1.A: What is the number of warps per block?

We know there's at 32 Threads per warp. The kernel is called<<<gridDim, blockDim>>> So N=1024 -> <<< 9, 128>>> So 128 threads/block means 128/32 == 
	4 Warps per block!

1.B: What is the number warps in the grid?

Grid consists of 9 blocks of 128 threads or 8 blocks of 4 warps thus 
	32 warps per grid

1.C: For the statement on line 4: b[i] = a[i] + 1;
1.C.I: How many warps in the grid are active?
	Conditional is: if (threadIdx.x < 40 || threadIdx.x >= 104)
	Each block is 128 threads, Line 4 is relevant for Threads 0-39 and 104-127
	Warp 0: 0-31
	Warp 1: 32-63
	Warp 2: 64-95
	Warp 3: 96-127
	SO line 4 covers All of warp 0, and part of warps 1 and 3.
	Across the grid it would be 8 fully active warps and 16 partially active warps.
1.C.II: How many warps in the Grid are Divergent?
	That would be two warps, Warps 1 and 3 as they contain some threads covered/not covered by the preceding conditional.
	16 warps are divergent
1.C.III: What is SIMD eff (%) of warp 0 in block 0?
	100%
1.C.IV: "" Warp 1 of Block o?
	Warp 1 = 32-63, only 32-39 are covered so 8/32 == 25%?
1.C.V: "" Warp 3 of Block 0?
	Warp 3 = 96-127 only 104-127 are covered so 24/32 == 75%?

1.D. For Line 07
1.D.I: How many warps in the Grid are active?
	Preceding conditional: if (i % 2 == 0)
		BlockIdx.x: 0, 1, 2...8
		BlockDim.x: 128
		ThreadIdx.x: 0, 1, 2...127
		Ex: 0*128+1-127
			1*128+1-127
		All warps in the Grid are Active (32)
1.D.II: How many warps in the Grid are divergent?
		All warps are divergent! (32)


*/

//Question 2: For a vector addition assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads, how many threads are in the grid?
/*

2000 Elements-> 2000 threads, 512 threads/block, 2000%512==48, so 2048 for 4 blocks across the grid.

2048 threads in the grid

*/
//Question 3: based off #2, how many warps do you expec to have divergence?
/*
Well, we're fine for blocks 1,2,3 (threads 0-511, 512-1023, 1024-1535), only have divergence on one warp.
*/
//Question 4: 1.9-3.0 = 1.1 ms waiting
//Quesiton 5: No because the purpose of the __syncthreads() is specifically for blocks, its whole purpose is to ensure all threads in the block are synced.
//Question 6: If a CUDA device's SM can take up to 1536 Threads  and up to 4 Blocks, which of the following configs would result in the most number of threads in the SM?
// C: 512 Thread Blocks
//Question 7: Assume a device with up to 64 Blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM is possible then indicate the occupancy level.
/*
 8 Blocks, 128 Threads -- Good, 50%
 16 64 -- Good, 50%
 32 32 -- Good, 50%
 64 32 -- Good,100%
 32 64 -- Good 100%
*/
//Question 8: Consider a GPU with 2048 threads per SM, 32 Blocks per SM, and 65,536 Registers per SM, for each of the following configurations specify if the kernel can achieve full occupancy and if not explain why.
/*

Kernel with 128 Threads/Block and 30 regs/thread -> 32 Blocks, 4096 Threads (>>2048) and 122K registers, Lim fac is too many threads/block, should cut to 64 threads/block
"" 32 threads per block and 29 registers/thread -> No limfacs, doesn't achieve full occupancy though.
"" 256 Threads and 34 Regsiter/thread, 8 blocks of total 2048 threads, limfac is too many registers per thread 2048*34 = 69K
*/

//Question 9: Student multiplies 1024*1024 matrix (1,048,576 elements) with 32*32 thread blocks (1024 threads/block) in his kernel but his device permits 512 th/bl  and 8 bl/SM
//What's wrong? His kernel blocks have double the permitted number of threads by his CUDA Device.
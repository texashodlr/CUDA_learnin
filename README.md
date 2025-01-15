# CUDA_learnin
Just learnin' a lil' cuda!

Basically this repo is a documentation of my journey to becoming a CUDA Kernel Engineer and building AI and serving my country, the usual.

The laptop I'm building most of this on is GTX4070-Laptop edition.

It's got 36 Streaming Multiprocessors (SMs) with 128 Cores/SM -- ADA Lovelace Arch.

Interesting notes from this SO Article: [https://stackoverflow.com/questions/2207171/how-much-is-run-concurrently-on-a-gpu-given-its-numbers-of-sms-and-sps/2213744#2213744]

1. Executing a functional/kernel on the GPU means assigning some amount of WORK as grid of block of threads.
2. Threads are the finest granularity and exist in blocks with specific(unique) identifiers (ThreadIdx.x/y/z).
3. A block is a group of threads (BlockDIM/BlockIdx) which execute in a batch and threads can communicate with eachother intra-block (via shared memory)
4. A Warp consists of 32 threads -- EX: 128 threads in a block, threads 0-31==warp1, 32-63==warp2...
  1. A warp is a group of 32 threads that execute in lockstep on an SM (Streaming Multiprocessor).
  2. All threads in a warp execute the same instruction at the same time (SIMD-like behavior).
  3. Potentially dealing with Warp Divergence
5. Threads within a warp fetch memory together (be concious and try to minimize, could get to a single mem-tx) // Also could lead to bank-conflicts
6. Blocks are launched on SMs, 1-N Blocks to an SM and once that block(s) retires then the SM gets another block from the Minecraft God
7. Each threadblock in the grid associated with a kernel launch is assigned to one SM (when the SM has a free slot). 
	The SM then "unpacks" the threadblock into warps, and schedules warp instructions on the SM internal resources 
    (e.g. "cores", and special function units), as those resources become available.

In the case of the add.cu code:
1. We've got this line -> N = 1 << 20 (1,048,576 elements):
2. This line: -> numBlocks = (N + blockSize - 1) / blockSize = 4096
2.1. Thus total blocks in our grid = 4096
3. This line: -> blockSize = 256 threads.
3.1 Total threads = 1 Grid x 4096 Blocks x 256 Threas == 1,048,576 Threads (Elements!). 
4. For Warps, if each block has 256 threads, 32 warps in a thread we're at 8 warps/block
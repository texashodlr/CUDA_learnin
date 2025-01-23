//Notes from Chapter 4
/*

__synchthreads();

In correct use == threads waiting forever, deadlock

A block can being execution only when the runtime system has secured all the resources needed by all threads in the block to complete execution
	This ensures the time proximity of all threads ina  block and prevents an excessive or even indefinite waiting time during the barrier sync

Each block can execute in any order relative to the other blocks its just that the threads inside the block must execute on the same SM

Blocks are partitioned into warps for thread scheduling, 32 threads in a warp and so forth.
	In general warp 'n' starts with thread 32*n and ends with thread 32*(n+1)-1

Warps < 32 threads are padded with inactive threads

SMs execute in a SIMD model == Single Insn Multiple Thread

Execution path == control flow

__syncwarp()

zero overhead thread scheduling
Thread oversubscription for SMs

Built in CUDA C mechanisms:
	int devCount;
	cudaGetDeviceCount(&devCount);

	cudaDeviceProp devProp;
	for(unsigned int i=0; i< devCount; i++){
		cudaGetDeviceProperties(&devProp, i);
		//Decide if the device has sufficient resources/capes
	}

	devProp.maxThreadsPerBlock -- gives max number fo threads allowed in a block in the queried device
	devProp.multiProcessorCount -- gives number of SMs in the device
	devProp.clockRate -- Clock freq
	devProp.maxThreadsDim[0,1,2] -- max threads in x/y/z dimensions
	devProp.maxGridSize[0,1,2] -- max blocks in x/y/z dimensions
	devProp.regsPerBlock -- number of register that are avail in each SM
	devProp.warpSize -- for size of warps

*/

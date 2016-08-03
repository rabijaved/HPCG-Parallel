#include "cuDProd.h"


__global__ void kernelDProduct(int N, int item_per_thread, 
double * __restrict__ a_vec, 
double * __restrict__ b_vec,
double * __restrict__ output) {

	extern __shared__ double partial_dot[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int lid = threadIdx.x;
	int group_size = blockDim.x;

	int offset = gid *item_per_thread;
	int x;
	partial_dot[lid] = 0;
	for (x = offset ; x < offset +  item_per_thread; x++){
		if(x < N){
			partial_dot[lid] += a_vec[x] * b_vec[x];
		}	
	}
	__syncthreads(); 

	for(int i = group_size/2; i>0; i >>= 1) {
		if(lid < i) {
			partial_dot[lid] += partial_dot[lid + i];
		}
		__syncthreads();
	}



	if(lid == 0) {
		
		output[blockIdx.x] = partial_dot[0];

	}

}

//get optimal block size
int getDProdBlockSize(int n,int minGridSize, int blockSize){
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelDProduct, 0, n); 

	return blockSize;

}


double runDotProduct(int n,
double * xMatrix,
int xMatrixSize,
double * yMatrix,
int yMatrixSize,
double * dprod_output,
int outMatrixSize,
double * xMatrix_d,
double * yMatrix_d,
double * dprod_output_d,
int item_per_thread,
int blockSize,
int numBlocks){


	int x;
	double my_res= 0;
	
	//copy arrays to host
	cudaMemcpy( xMatrix_d, xMatrix, sizeof(double)*xMatrixSize, cudaMemcpyHostToDevice );
	cudaMemcpy( yMatrix_d, yMatrix, sizeof(double)*yMatrixSize, cudaMemcpyHostToDevice );

	//blockSize = blockSize/item_per_thread;

	//launch kernel
	kernelDProduct<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(n, item_per_thread,
	xMatrix_d, yMatrix_d, dprod_output_d);

	cudaMemcpy(dprod_output, dprod_output_d, sizeof(double)*numBlocks, cudaMemcpyDeviceToHost );

	//reduction
	for( x = 0;x < numBlocks; x++){
	
		my_res += dprod_output[x];
	}

	return my_res;

}

#include "cuRestriction.h"

__global__ void kernelRestriction(int nc,int item_per_thread,
double * __restrict__ axfMatrix,
double * __restrict__ rfMatrix,
double * __restrict__ rcMatrix,
int * __restrict__ f2cMatrix
)               
{
	unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gid * item_per_thread;
	int x;
	for (x = offset ; x < offset +  item_per_thread; x++){
		if(x < nc){
			
			rcMatrix[x] = rfMatrix[f2cMatrix[x]] - axfMatrix[f2cMatrix[x]];

		}

	}
	
}


//get optimal block size
int getRestrictionBlockSize(int n,int minGridSize, int blockSize){

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelRestriction, 0, n); 

	return blockSize;

}




int runRestriction(int nc,
double * rfMatrix,
int rfMatrixSize,
double * rcMatrix,
int rcMatrixSize,
double * axfMatrix,
int axfMatrixSize,
double * axfMatrix_d,
double * rfMatrix_d,
double * rcMatrix_d,
int * f2cMatrix_d,
int item_per_thread,
int blockSize,
int numBlocks){


	cudaMemcpy( rfMatrix_d, rfMatrix, sizeof(double)*rfMatrixSize, cudaMemcpyHostToDevice );

	cudaMemcpy( axfMatrix_d, axfMatrix, sizeof(double)*axfMatrixSize, cudaMemcpyHostToDevice );

	kernelRestriction<<<numBlocks, blockSize>>>(nc, item_per_thread, axfMatrix_d, rfMatrix_d, rcMatrix_d, f2cMatrix_d);

	cudaMemcpy(rcMatrix, rcMatrix_d, sizeof(double)*rcMatrixSize, cudaMemcpyDeviceToHost );


	return 0;

}

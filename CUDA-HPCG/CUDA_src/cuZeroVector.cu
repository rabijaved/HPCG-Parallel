#include "cuZeroVector.h"


__global__ void kernelZeroVector(int n, int item_per_thread,
double * __restrict__ xMatrix)
{

	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
	int x = 0;
	int offset = row *item_per_thread;
	
	for(x = offset ; x< offset + item_per_thread; x++){
		if(x < n) {

			xMatrix[x] = (double4)0.0f;

		}


	}

}

//get optimal block size
int getZeroVectorBlockSize(int n,int minGridSize, int blockSize){

 cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelZeroVector, 0, n); 
 
return blockSize;

}


int runZeroVector(int n,
		double * xMatrix,
		double * xMatrix_d,
	    int item_per_thread,
		int blockSize,
		int numBlocks){



	 kernelZeroVector<<<numBlocks, blockSize>>>(n, item_per_thread, xMatrix_d);

	 cudaMemcpy(xMatrix, xMatrix_d, sizeof(double)*n, cudaMemcpyDeviceToHost );




}
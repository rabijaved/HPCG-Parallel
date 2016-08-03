#include "cuWAXPBY.h"



__global__ void kernelWAXPBY(int n,int items_per_thread,
		  double alpha,
		  double beta,
		  double * __restrict__ xMatrix,
		  double * __restrict__ yMatrix,
		  double * __restrict__ wMatrix
		)               
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	int offset= i *items_per_thread;
	int x;
	
	for(x = offset; x < offset + items_per_thread; x++){
		if( i < n ){
			if (alpha==1.0) {
			    wMatrix[x] = xMatrix[x] + beta * yMatrix[x];
			} else if (beta==1.0) {
			    wMatrix[x] = alpha * xMatrix[x] + yMatrix[x];
			} else  {
			    wMatrix[x] = alpha * xMatrix[x] + beta * yMatrix[x];
			}
			
		}
	}
}


//get optimal block size
int getWAXPBYBlockSize(int n,int minGridSize, int blockSize){

 cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelWAXPBY, 0, n); 

return blockSize;

}


int runWAXPBY(int n,double alpha,double beta,
		double * xMatrix,
		double xMatrixSize,
		double * yMatrix,
		double yMatrixSize,
		double * wMatrix,
		double wMatrixSize,
		double * xMatrix_d,
		double * yMatrix_d,
		double * wMatrix_d,
	        int item_per_thread,
		int blockSize,
		int numBlocks
){

	 cudaMemcpy( xMatrix_d, xMatrix, sizeof(double)*xMatrixSize, cudaMemcpyHostToDevice );
	 cudaMemcpy( yMatrix_d, yMatrix, sizeof(double)*yMatrixSize, cudaMemcpyHostToDevice );

	 kernelWAXPBY<<<numBlocks, blockSize>>>(n, item_per_thread, alpha, beta, xMatrix_d, yMatrix_d, wMatrix_d);

	 cudaMemcpy(wMatrix, wMatrix_d, sizeof(double)*wMatrixSize, cudaMemcpyDeviceToHost );

	 return 0;

}


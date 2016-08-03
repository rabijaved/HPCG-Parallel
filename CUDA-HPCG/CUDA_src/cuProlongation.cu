#include "cuProlongation.h"


__global__ void kernelProlongation(int n,int item_per_thread,
double * __restrict__ xfMatrix,
double * __restrict__ xcMatrix,
int * __restrict__ f2cMatrix
)               
{
	unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gid * item_per_thread;
	int x;
	for (x = offset ; x < offset +  item_per_thread; x++){
		if (x<n){ 
			xfMatrix[f2cMatrix[x]] = xfMatrix[f2cMatrix[x]] + xcMatrix[x]; 
			//printf( "Prolongation output: %f\n",xfMatrix[f2cMatrix[i]]); 
		}
	}
	

}

//get optimal block size
int getProlongationBlockSize(int n,int minGridSize, int blockSize){

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelProlongation, 0, n); 

	return blockSize;
}

int runProlongation(int nc,
double * xfMatrix,
int xfMatrixSize,
double * xcMatrix,
int xcMatrixSize,
double * xfMatrix_d,
double * xcMatrix_d,
int * f2cMatrix_d,
int item_per_thread,
int blockSize,
int numBlocks
){


	cudaMemcpy( xcMatrix_d, xcMatrix, sizeof(double)*xcMatrixSize, cudaMemcpyHostToDevice );

	cudaMemcpy( xfMatrix_d, xfMatrix, sizeof(double)*xfMatrixSize, cudaMemcpyHostToDevice );

	kernelProlongation<<<numBlocks, blockSize>>>(nc, item_per_thread, xfMatrix_d, xcMatrix_d, f2cMatrix_d);

	cudaMemcpy(xfMatrix, xfMatrix_d, sizeof(double)*xfMatrixSize, cudaMemcpyDeviceToHost );

	return 0;
}

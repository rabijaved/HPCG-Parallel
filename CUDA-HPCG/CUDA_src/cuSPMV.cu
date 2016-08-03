#include "cuSPMV.h"
#include <cuda_runtime.h>
 


__global__ void spmvKernel(int num_rows, int item_per_thread,
		char * __restrict__ AnonzerosInRow,
		int * __restrict__ AmtxIndL,
		double * __restrict__ AmatrixValues,
		double * __restrict__ xMatrix,
		double * __restrict__ yMatrix)
{

	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int x;
	int i;
	int offset = row *item_per_thread;
        double sum;	
	for(x = offset ; x< offset + item_per_thread; x++){
		if(x < num_rows) {
			sum = 0;
			for (i = 0; i < AnonzerosInRow[x]; i++){
				sum +=	AmatrixValues[x * 27 + i] * xMatrix[ AmtxIndL[x * 27 + i] ];	
			}	
		
		yMatrix[x] = sum;
	}
}
}


//---------------------------------------------MAIN---------------------------------------------

//get optimal block size
int getSPMVBlockSize(int n,int minGridSize, int blockSize){

 cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, spmvKernel, 0, n); 

 
return blockSize;

}




int runSPMV(int localNumberOfRows,
		int numberOfNonzerosPerRow,
		double * xMatrix,
		double xMatrixSize,
		double * yMatrix,
		double yMatrixSize,
		char * AnonzerosInRow_d,
		int *AmtxIndL_d,
		double *AmatrixValues_d,
		double* xMatrix_d,
		double *yMatrix_d,
		int item_per_thread,
		int blockSize,
		int numBlocks
		){


	 cudaMemcpy( xMatrix_d, xMatrix, sizeof(double)*xMatrixSize, cudaMemcpyHostToDevice );

	 //blockSize = blockSize/item_per_thread;
	 
	 spmvKernel<<<numBlocks, blockSize>>>(localNumberOfRows, item_per_thread, AnonzerosInRow_d, AmtxIndL_d, AmatrixValues_d, xMatrix_d, yMatrix_d);

	 cudaMemcpy(yMatrix, yMatrix_d, sizeof(double)*yMatrixSize, cudaMemcpyDeviceToHost );



    return 0;

}



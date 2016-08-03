#ifndef OCLSPMV_H
#define OCLSPMV_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>


#include <cuda.h>

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
		);


int getSPMVBlockSize(int n,int minGridSize, int blockSize);

#endif


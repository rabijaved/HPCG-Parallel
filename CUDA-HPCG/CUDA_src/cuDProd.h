#ifndef CUDPROD_H
#define CUDPROD_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda.h>

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
		  int numBlocks);


int getDProdBlockSize(int n,int minGridSize, int blockSize);

#endif


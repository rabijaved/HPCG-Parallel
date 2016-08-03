#ifndef CUWAXPBY_H
#define CUWAXPBY_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>


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
);


int getWAXPBYBlockSize(int n,int minGridSize, int blockSize);

#endif


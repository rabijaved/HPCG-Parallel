#ifndef CUPROLONGATION_H
#define CUPROLONGATION_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>

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
);


int getProlongationBlockSize(int n,int minGridSize, int blockSize);

#endif


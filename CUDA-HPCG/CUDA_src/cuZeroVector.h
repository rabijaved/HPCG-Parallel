#ifndef OCLZERO_H
#define OCLZERO_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>


int runZeroVector(int n,
		double * xMatrix,
		double * xMatrix_d,
	    int item_per_thread,
		int blockSize,
		int numBlocks);


int getZeroVectorBlockSize(int n,int minGridSize, int blockSize);


#endif


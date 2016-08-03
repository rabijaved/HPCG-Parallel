#ifndef OCLREST_H
#define OCLREST_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>


int runRestriction(int nc,
                double * rfMatrix,
                int rfMatrixSize,
                double * rcMatrix,
                int rcMatrixSize,
                double * axfMatrixMatrix,
                int axfMatrixMatrixSize,
                double * axfMatrix_d,
                double * rfMatrix_d,
                double * rcMatrix_d,
                int * f2cMatrix_d,
                int item_per_thread,
                int blockSize,
                int numBlocks);

int getRestrictionBlockSize(int n,int minGridSize, int blockSize);


#endif


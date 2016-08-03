#ifndef GETDEV_H
#define GETDEV_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <cuda.h>


void printDevProp(cudaDeviceProp devProp);
int cuGetDevice();


#endif


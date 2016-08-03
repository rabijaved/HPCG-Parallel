#ifndef OCLWAXPBY_H
#define OCLWAXPBY_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "buildKernel.h"
#include "createDevice.h"

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


int runWAXPBY(int n,double alpha,double beta,
		double * xMatrix,
		double xMatrixSize,
		double * yMatrix,
		double yMatrixSize,
		double * wMatrix,
		double wMatrixSize,
		cl_kernel my_kernel,
		cl_mem output_buffer,
		cl_mem xMatrix_buff,
		cl_mem yMatrix_buff,
		cl_command_queue queue,
		int item_per_thread
);

#ifdef __cplusplus
}
#endif

#endif


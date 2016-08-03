#ifndef OCLRESTRICTION_H
#define OCLRESTRICTION_H

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


int runRestriction(int nc,
		double * axfMatrix,
		int axfMatrixSize,
		double * rfMatrix,
		int rfMatrixSize,
		double * rcMatrix,
		int rcMatrixSize,
		int * f2cMatrix,
		int f2cMatrixSize,
		cl_kernel my_kernel,
		cl_mem axfMatrix_buff,
		cl_mem rfMatrix_buff,
		cl_mem rcMatrix_buff,
		cl_mem f2cMatrix_buff,
		cl_command_queue queue,
		int item_per_thread);


#ifdef __cplusplus
}
#endif

#endif


#ifndef OCLSYMGS_H
#define OCLSYMGS_H

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


int runSYMGS(int localNumberOfRows,
		double * xMatrix,
		double xMatrixSize,
		double * bMatrix,
		double bMatrixSize,
		cl_kernel my_kernel,
		cl_mem output_buffer,
		cl_mem xMatrix_buff,
		cl_mem bMatrix_buff,
		cl_command_queue queue
		);



#ifdef __cplusplus
}
#endif

#endif




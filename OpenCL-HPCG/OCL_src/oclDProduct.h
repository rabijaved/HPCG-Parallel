#ifndef OCLDPRODUCT_H
#define OCLDPRODUCT_H

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


double runDotProduct(int n,
	      double * xMatrix,
	      int xMatrixSize,
	      double * yMatrix,
	      int yMatrixSize,
	      double * dprod_temp_output,
	      int rMatrixSize,
	      cl_kernel dot_kernel,
		  cl_mem xMatrix_buff,
		  cl_mem yMatrix_buff,
		  cl_mem dprod_output_buff,
		  cl_command_queue queue,
		  int item_per_thread);


#ifdef __cplusplus
}
#endif

#endif


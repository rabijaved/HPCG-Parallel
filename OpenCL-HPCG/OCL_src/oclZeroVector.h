#ifndef OCLZERO_H
#define OCLZERO_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>


#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


int runZeroVector(int n,
		double * xMatrix,
		cl_kernel my_kernel,
		cl_mem xMatrix_buff,
		cl_command_queue queue,
		int item_per_thread);



#ifdef __cplusplus
}
#endif

#endif


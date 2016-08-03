#include "oclZeroVector.h"



int runZeroVector(int n,
		double * xMatrix,
		cl_kernel my_kernel,
		cl_mem xMatrix_buff,
		cl_command_queue queue,
		int item_per_thread){


	   size_t global_size;
	   cl_int err;

	   global_size = n/(4*item_per_thread);

	   err = clEnqueueNDRangeKernel(queue, my_kernel, 1, NULL, &global_size,
			   NULL, 0, NULL, NULL);

	   err |= clEnqueueReadBuffer(queue, xMatrix_buff, CL_TRUE, 0,
			   sizeof(double)*n, xMatrix, 0, NULL, NULL);

	   if(err < 0) {
	      perror("Error occured in zero vector");
	      exit(1);
	   }

return 0;

}

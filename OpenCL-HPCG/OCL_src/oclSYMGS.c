#include "oclSYMGS.h"


//---------------------------------------------MAIN---------------------------------------------


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
		){


	   size_t global_size;
	   cl_int err;


	   err |= clEnqueueWriteBuffer(queue, xMatrix_buff, CL_FALSE,
	     0, sizeof(double)*xMatrixSize, xMatrix, 0, NULL, NULL);
	   if(err < 0) { perror("Couldn't copy buffer"); exit(1);};


	   err |= clEnqueueWriteBuffer(queue, bMatrix_buff, CL_FALSE,
	     0, sizeof(double)*bMatrixSize, bMatrix, 0, NULL, NULL);
	   if(err < 0) { perror("Couldn't copy buffer"); exit(1);};


	   global_size = localNumberOfRows;

	   err = clEnqueueNDRangeKernel(queue, my_kernel, 1, NULL, &global_size,
			   NULL, 0, NULL, NULL);

	   err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
			   sizeof(double)*xMatrixSize, xMatrix, 0, NULL, NULL);


	   return 0;

}

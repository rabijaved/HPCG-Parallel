#include "oclRestriction.h"


//---------------------------------------------MAIN---------------------------------------------


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
		int item_per_thread){


	   size_t global_size;
	   cl_int err;


	   err = clEnqueueWriteBuffer(queue, axfMatrix_buff, CL_FALSE,
	     0, sizeof(double)*axfMatrixSize, axfMatrix, 0, NULL, NULL);

	   err |= clEnqueueWriteBuffer(queue, rfMatrix_buff, CL_FALSE,
	     0, sizeof(double)*rfMatrixSize, rfMatrix, 0, NULL, NULL);

	   if(err < 0) { perror("Couldn't copy buffer"); exit(1);};


	   global_size = nc/item_per_thread;
	   err = clEnqueueNDRangeKernel(queue, my_kernel, 1, NULL, &global_size,
	         NULL, 0, NULL, NULL);

	   if(err < 0) {
	      perror("Couldn't enqueue the kernel");
	      exit(1);
	   }

	   err = clEnqueueReadBuffer(queue, rcMatrix_buff, CL_TRUE, 0,
			   sizeof(double)*rcMatrixSize, rcMatrix, 0, NULL, NULL);
	   if(err < 0) {
	      perror("Couldn't read the buffer");
	      exit(1);
	   }


	   return 0;

}



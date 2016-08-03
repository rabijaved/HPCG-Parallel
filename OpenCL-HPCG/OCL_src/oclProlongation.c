#include "oclProlongation.h"



int runProlongation(int nc,
		double * xfMatrix,
		int xfMatrixSize,
		double * xcMatrix,
		int xcMatrixSize,
		int * f2cMatrix,
		int f2cMatrixSize,
		cl_kernel my_kernel,
		cl_mem xfMatrix_buff,
		cl_mem xcMatrix_buff,
		cl_mem f2cMatrix_buff,
		cl_command_queue queue,
		int item_per_thread){

	   size_t global_size;
	   cl_int err;


	   err = clEnqueueWriteBuffer(queue, xfMatrix_buff, CL_FALSE,
	     0, sizeof(double)*xfMatrixSize, xfMatrix, 0, NULL, NULL);

	   err |= clEnqueueWriteBuffer(queue, xcMatrix_buff, CL_FALSE,
	     0, sizeof(double)*xcMatrixSize, xcMatrix, 0, NULL, NULL);

	   if(err < 0) { perror("Couldn't copy buffer"); exit(1);};

	   global_size = nc/(item_per_thread);

	   err = clEnqueueNDRangeKernel(queue, my_kernel, 1, NULL, &global_size,
	         NULL, 0, NULL, NULL);

	   if(err < 0) {perror("Couldn't enqueue prolongation kernel"); exit(1); }

	   err = clEnqueueReadBuffer(queue, xfMatrix_buff, CL_TRUE, 0,
			   sizeof(double)*xfMatrixSize, xfMatrix, 0, NULL, NULL);
	   if(err < 0) {perror("Couldn't read output buffer"); exit(1); }



	   return 0;

}



#include "oclWAXPBY.h"



//---------------------------------------------MAIN---------------------------------------------


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
){


	   size_t global_size;
	   cl_int err;


	   err = clEnqueueWriteBuffer(queue, xMatrix_buff, CL_TRUE,
	     0, sizeof(double)*xMatrixSize, xMatrix, 0, NULL, NULL);

	   err |= clEnqueueWriteBuffer(queue, yMatrix_buff, CL_TRUE,
	     0, sizeof(double)*yMatrixSize, yMatrix, 0, NULL, NULL);

	   if(err < 0) { perror("Couldn't copy buffer"); exit(1);};


	   err = clSetKernelArg(my_kernel, 2, sizeof(double), &alpha);
	   err |= clSetKernelArg(my_kernel, 3, sizeof(double), &beta);


	   global_size = n/(item_per_thread*8);

	   err = clEnqueueNDRangeKernel(queue, my_kernel, 1, NULL, &global_size,
	         NULL, 0, NULL, NULL);
	   if(err < 0) {
	      perror("Couldn't enqueue the kernel");
	      exit(1);
	   }


	   err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
			   sizeof(double)*wMatrixSize, wMatrix, 0, NULL, NULL);
	   if(err < 0) {
	      perror("Couldn't read the buffer");
	      exit(1);
	   }


	   return 0;

}



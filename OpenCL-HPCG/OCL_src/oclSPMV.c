#include "oclSPMV.h"


//---------------------------------------------MAIN---------------------------------------------


int runSPMV(int localNumberOfRows,
		double * xMatrix,
		double xMatrixSize,
		double * yMatrix,
		double yMatrixSize,
		cl_kernel my_kernel,
		cl_mem output_buffer,
		cl_mem xMatrix_buff,
		cl_command_queue queue,
		int item_per_thread
		){


	   size_t global_size;
	   cl_int err = 0;

	   /* Timing Stuff
	   cl_event event;
	   struct timeval start, finish;
	   cl_ulong time_start, time_end;
	   double total_time;
	   gettimeofday(&start, NULL);


	   clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
	            sizeof(max_local_size), &max_local_size, NULL);

	   */

	   err |= clEnqueueWriteBuffer(queue, xMatrix_buff, CL_FALSE,
	     0, sizeof(double)*xMatrixSize, xMatrix, 0, NULL, NULL);
	   if(err < 0) { perror("Couldn't copy buffer"); exit(1);};


	   //clFinish(queue);

	   global_size = localNumberOfRows/item_per_thread;

	   err = clEnqueueNDRangeKernel(queue, my_kernel, 1, NULL, &global_size,
			   NULL, 0, NULL, NULL);

	   err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
			   sizeof(double)*yMatrixSize, yMatrix, 0, NULL, NULL);

	   /*
	   clWaitForEvents(1 , &event);

	   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	   total_time = time_end - time_start;

	   gettimeofday(&finish, NULL);


	   printf("on device = %0.6f s total %u.%06u s\n", (total_time / 1000000000.0), (unsigned int)(finish.tv_sec - start.tv_sec),
		         (unsigned int)(finish.tv_usec - start.tv_usec) );

		*/
	   //clFlush(queue);


	   return 0;

}

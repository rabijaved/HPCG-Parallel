#include "oclDProduct.h"

//---------------------------------------------MAIN---------------------------------------------

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
	  int item_per_thread) {


   size_t global_size;
   cl_int err;
   double my_res = 0;

/*initialize here for testing only
   int fg=0;


   double dot_check = 0;
   for(int ii=0; ii<4096; ii++) {
      dot_check += xMatrix[ii] * yMatrix[ii];
   }

   //clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_local_size), &max_local_size, NULL);
*/
   /* my command queue */




   err = clEnqueueWriteBuffer(queue, xMatrix_buff, CL_FALSE,
     0, sizeof(double)*xMatrixSize, xMatrix, 0, NULL, NULL);

   err |= clEnqueueWriteBuffer(queue, yMatrix_buff, CL_FALSE,
     0, sizeof(double)*yMatrixSize, yMatrix, 0, NULL, NULL);
/*
   err |= clEnqueueWriteBuffer(queue, rMatrix_buff, CL_FALSE,
     0, sizeof(double)*rMatrixSize, rMatrix, 0, NULL, NULL);
*/
   if(err < 0) {
      perror("Couldn't copy buffer");
      exit(1);
   };

   cl_event events;
   global_size = n/(4*item_per_thread);
   //max_local_size = 128;
   err = clEnqueueNDRangeKernel(queue, dot_kernel, 1, NULL, &global_size,
         NULL, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the dot product kernel");
      exit(1);
   }
   /* Read output buffer
   err = clEnqueueReadBuffer(queue, dprod_output_buff, CL_TRUE, 0,
		   sizeof(double)*rMatrixSize, dprod_temp_output, 0, NULL, NULL);

   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }*
*/

   dprod_temp_output = (double*)clEnqueueMapBuffer(queue, dprod_output_buff,
                                               CL_TRUE, CL_MAP_READ, 0,
											   sizeof(double)*rMatrixSize,
                                               0, NULL, NULL, &err );
   if(err < 0) {perror("Couldn't create map buffer");exit(1);};


   for( int fg = 1;fg < dprod_temp_output[0]+1; fg++){
      my_res += dprod_temp_output[fg];
      dprod_temp_output[fg] = 0;
   }

   dprod_temp_output[0] = 0;

   //printf("expected value: %f actual val %f\n",dot_check,my_res);

   clEnqueueUnmapMemObject( queue,dprod_output_buff,  dprod_temp_output,0 , NULL, NULL);
   return my_res;
}

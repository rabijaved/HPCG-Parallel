#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void kernelWAXPBY(int n,int items_per_thread,
		  double alpha,
		  double beta,
		__global double8 * xMatrix,
		__global double8 * yMatrix,
		__global double8 * wMatrix
		)               
{
	unsigned int i = get_global_id(0);
	int offset= i *items_per_thread;
	int x;
	
	for(x = offset; x < offset + items_per_thread; x++){
		if( i < n ){
			if (alpha==1.0) {
			    wMatrix[x] = xMatrix[x] + beta * yMatrix[x];
			} else if (beta==1.0) {
			    wMatrix[x] = alpha * xMatrix[x] + yMatrix[x];
			} else  {
			    wMatrix[x] = alpha * xMatrix[x] + beta * yMatrix[x];
			}
			
		}
	}
}


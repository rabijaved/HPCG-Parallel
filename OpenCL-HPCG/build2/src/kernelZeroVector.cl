#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif


__kernel void kernelZeroVector(int n, int item_per_thread,
__global double4 * xMatrix)
{

	unsigned int row = get_global_id(0);
	int x = 0;
	int offset = row *item_per_thread;
	
	for(x = offset ; x< offset + item_per_thread; x++){
		if(x < n) {

			xMatrix[x] = (double4)0.0f;

		}


	}

}
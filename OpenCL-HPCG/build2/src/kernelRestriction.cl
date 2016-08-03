#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void kernelRestriction(int nc,int item_per_thread,
__global double * axfMatrix,
__global double * rfMatrix,
__global double * rcMatrix,
__global int * f2cMatrix
)               
{
	unsigned int gid = get_global_id(0);
	int offset = gid * item_per_thread;
	int x;
	for (x = offset ; x < offset +  item_per_thread; x++){
		if(x < nc){
			
			rcMatrix[x] = rfMatrix[f2cMatrix[x]] - axfMatrix[f2cMatrix[x]];

		}

	}
	
}


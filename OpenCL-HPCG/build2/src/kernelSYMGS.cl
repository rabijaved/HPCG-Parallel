#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif


__kernel void kernelSYMGS(int num_rows,
		int stride,
		__global char * AnonzerosInRow,         
		__global int * AmtxIndL,        		
		__global double * Adiagonal,        	
		__global double * AmatrixValues,         
		__global double * xMatrix,               
		__global double * bMatrix)               
{


	unsigned int row_index = get_global_id(0);
	int j = 0;
	double sum = bMatrix[row_index];
	if(row_index < num_rows) {
		
		for (j=0; j< AnonzerosInRow[row_index]; j+= stride) {
			
			sum -= AmatrixValues[j * 27 + row_index] * xMatrix[AmtxIndL[j * 27 + row_index]];

		}

		 sum += xMatrix[row_index]*Adiagonal[row_index];
		 xMatrix[row_index] = sum/Adiagonal[row_index];
	}
}
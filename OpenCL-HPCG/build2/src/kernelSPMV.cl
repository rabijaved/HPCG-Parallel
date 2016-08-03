#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void kernelSPMV(int num_rows,           //A.localNumberOfRows
int item_per_thread,
__global char * AnonzerosInRow,          //A.nonzerosInRow[i]
__global int * AmtxIndL,        		 //A.mtxIndL[i];
__global double * AmatrixValues,         //A.matrixValues[i]
__global double * xMatrix,               //x.values
__global double * yMatrix)               //y.values
{


	unsigned int row = get_global_id(0);
	int x,i,row_end;
	double sum;
	int offset = row *item_per_thread;
	
	for(x = offset ; x< offset + item_per_thread; x++){
		if(x < num_rows) {
			sum = 0;
			row_end = AnonzerosInRow[x];
			
			for (i = 0; i < row_end; i++){
				
				sum += AmatrixValues[x * 27 + i] * xMatrix[ AmtxIndL[x * 27 + i] ];
				
			}	
			
			yMatrix[x] = sum;

		}
		
	}

}







/*


	unsigned int row = get_global_id(0);
	int x = 0;
	int offset = row *item_per_thread;
	
for(x = offset ; x< offset + item_per_thread; x++){
	if(x < num_rows) {
		double sum = 0.0;
		int row_end = AnonzerosInRow[x];
		int i = 0;
				
		for (i = 0; i < row_end; i++){
		
			sum += AmatrixValues[i * num_rows + x] * xMatrix[ AmtxIndL[i * num_rows + x] ];
		
			//printf("row: %d col: %d mat val: %f  mat ind: %d x_val: %f\n", x,i, AmatrixValues[i * num_rows + x], AmtxIndL[i * num_rows + x],xMatrix[ AmtxIndL[i * num_rows + x]]);
		}	
		
		yMatrix[x] = sum;
		//printf("row: %f\n", sum);
	}
	
}
*/
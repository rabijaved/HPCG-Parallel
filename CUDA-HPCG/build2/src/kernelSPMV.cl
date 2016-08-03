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
	int x = 0;
	int offset = row *item_per_thread;
	
for(x = offset ; x< offset + item_per_thread; x++){
	if(x < num_rows) {
		double4 sum = (double4)0.0;
		int row_end = AnonzerosInRow[x];
		int i = 0;
				
		int residue = (row_end % 4);		
		for (i = 0; i < row_end - residue; i+=4){
		
			sum.x += AmatrixValues[x * 27 + i] * xMatrix[ AmtxIndL[x * 27 + i] ];
			sum.y += AmatrixValues[x * 27 + (i+1)] * xMatrix[ AmtxIndL[x * 27 + (i+1)] ];
			sum.z += AmatrixValues[x * 27 + (i+2)] * xMatrix[ AmtxIndL[x * 27 + (i+2)] ];
			sum.w += AmatrixValues[x * 27 + (i+3)] * xMatrix[ AmtxIndL[x * 27 + (i+3)] ];

		}	
		
		
		
	    if(residue == 3){
			sum.x += AmatrixValues[x * 27 + i] * xMatrix[ AmtxIndL[x * 27 + i] ];
			sum.y += AmatrixValues[x * 27 + (i+1)] * xMatrix[ AmtxIndL[x * 27 + (i+1)] ];
			sum.z += AmatrixValues[x * 27 + (i+2)] * xMatrix[ AmtxIndL[x * 27 + (i+2)] ];
		
		}
		else if(residue == 2){
		
			sum.x += AmatrixValues[x * 27 + i] * xMatrix[ AmtxIndL[x * 27 + i] ];
			sum.y += AmatrixValues[x * 27 + (i+1)] * xMatrix[ AmtxIndL[x * 27 + (i+1)] ];
		
		}
		else if (residue == 1){
		
			sum.x += AmatrixValues[x * 27 + i] * xMatrix[ AmtxIndL[x * 27 + i] ];
			
		}
		
				
		yMatrix[x] = dot(sum,(double4)1.0f);

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
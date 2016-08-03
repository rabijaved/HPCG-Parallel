#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel void kernelDProduct(int N, int item_per_thread, __global double4* a_vec, __global double4* b_vec,
__global double* output, __local double4* partial_dot) {



	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int group_size = get_local_size(0);

	int offset = gid *item_per_thread;
	int x;
	partial_dot[lid] = 0;
	for (x = offset ; x < offset +  item_per_thread; x++){
		if(x < N){
			partial_dot[lid] += a_vec[x] * b_vec[x];
		}	
	}
	barrier(CLK_LOCAL_MEM_FENCE);


	for(int i = group_size/2; i>0; i >>= 1) {
		if(lid < i) {
			partial_dot[lid] += partial_dot[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}



	if(lid == 0) {
		
		output[get_group_id(0)+1] = dot(partial_dot[0], (double4)(1.0f));
		//printf("\n%2.2v4hlf \n", partial_dot[0]);
		//printf("%f\n", output[get_group_id(0)+1]);
	}


	if(gid == 0 ) output[0] = get_num_groups(0);



}
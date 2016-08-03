
#ifndef HPCG_NOMPI
#include <mpi.h> // If this routine is not compiled with HPCG_NOMPI
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>

#include "hpcg.hpp"

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "GenerateCoarseProblem.hpp"
#include "SetupHalo.hpp"
#include "ExchangeHalo.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "CG.hpp"
#include "CG_ref.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "TestCG.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"
#include "FlattenMatrix.hpp"
#include "../OCL_src/oclSPMV.h"
#include "../OCL_src/oclWAXPBY.h"
#include "../OCL_src/oclDProduct.h"
#include "../OCL_src/buildKernel.h"
#include "../OCL_src/createDevice.h"

#define SPMV_PATH "../src/kernelSPMV.cl"
#define WAXPBY_PATH "../src/kernelWAXPBY.cl"
#define PROLONGATION_PATH "../src/kernelProlongation.cl"
#define DOTPRODUCT_PATH "../src/kernelDProduct.cl"
#define RESTRICTION_PATH "../src/kernelRestriction.cl"
#define ZEROVECTOR_PATH "../src/kernelZeroVector.cl"

#define SPMV_FUNCTION "kernelSPMV"
#define WAXPBY_FUNCTION "kernelWAXPBY"
#define PROLONGATION_FUNCTION "kernelPROLOG"
#define DOTPRODUCT_FUNCTION "kernelDProduct"
#define RESTRICTION_FUNCTION "kernelRestriction"
#define ZEROVECTOR_FUNCTION "kernelZeroVector"

#define HPCG_DEBUG
#define HPCG_DETAILED_DEBUG

//declare OpenCL constructs here

cl_mem spmv_output_buffer,spmv_nonzerosInRow_buff,spmv_mtxIndL_buff,spmv_matrixValues_buff,spmv_xMatrix_buff;
cl_mem waxpby_output_buffer, waxpby_xMatrix_buff, waxpby_yMatrix_buff;
cl_mem dprod_xMatrix_buff, dprod_yMatrix_buff, dprod_output_buff;
cl_mem xfMatrix_buff, xcMatrix_buff;
cl_mem restriction_axfMatrix_buff;
cl_mem zerovec_xMatrix_buff,f2cMatrix_buff;
cl_mem data_Z_values_buffer,A_mgData_Axf_values_buffer;
cl_device_id device;
cl_context context;
cl_program programZerovec, programSPMV,programWAXPBY,programProlongation, programDotProduct,programRestriction;
cl_kernel kernelZerovec, kernelSPMV,kernelWAXPBY,kernelProlongation,kernelDotProduct,kernelRestriction;
int zerovec_item_per_thread, spmv_item_per_thread, dprod_item_per_thread,waxpby_item_per_thread,prolongation_item_per_thread, restriction_item_per_thread;
cl_command_queue queue;
size_t max_local_size;
double *dprod_temp_output;



int main(int argc, char * argv[]) {

#ifndef HPCG_NOMPI
	MPI_Init(&argc, &argv);
#endif

	HPCG_Params params;

	HPCG_Init(&argc, &argv, params);

	int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID

#ifdef HPCG_DETAILED_DEBUG
	if (size < 100 && rank==0) HPCG_fout << "Process "<<rank<<" of "<<size<<" is alive with " << params.numThreads << " threads." <<endl;

	if (rank==0) {
		char c;
		std::cout << "Press key to continue"<< std::endl;
		std::cin.get(c);
	}
#ifndef HPCG_NOMPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif


	printf("\nmy rank %d, my size %d", rank,size);

	local_int_t nx,ny,nz;
	nx = (local_int_t)params.nx;
	ny = (local_int_t)params.ny;
	nz = (local_int_t)params.nz;
	int ierr = 0;  // Used to check return codes on function calls

	if(rank == 0){
		printf("\nInput Dimensions nx %d ny %d nz %d",nx,ny,nz);
		printf("\nStart problem setup phase\n");

	}


	// //////////////////////
	// Problem setup Phase //
	/////////////////////////

#ifdef HPCG_DEBUG
	double t1 = mytimer();
#endif

	// Construct the geometry and linear system
	Geometry * geom = new Geometry;
	GenerateGeometry(size, rank, params.numThreads, nx, ny, nz, geom);

	SparseMatrix A;
	InitializeSparseMatrix(A, geom);

	Vector b, x, xexact;
	GenerateProblem(A, &b, &x, &xexact);
	SetupHalo(A);
	int numberOfMgLevels = 4; // Number of levels including first
	SparseMatrix * curLevelMatrix = &A;
	for (int level = 1; level< numberOfMgLevels; ++level) {
		GenerateCoarseProblem(*curLevelMatrix);
		curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
	}


	CGData data;
	InitializeSparseCGData(A, data);


	// Use this array for collecting timing information
	std::vector< double > times(9,0.0);

	// Call user-tunable set up function.
	double t7 = mytimer(); OptimizeProblem(A, data, b, x, xexact); t7 = mytimer() - t7;
	times[7] = t7;




	//------------------------------initialize OCL Kernels---------------------------

	local_int_t numberOfNonzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil


	spmv_item_per_thread = 4;
	dprod_item_per_thread = 4;
	waxpby_item_per_thread =2;
	prolongation_item_per_thread = 2;
	restriction_item_per_thread = 2;
	zerovec_item_per_thread = 1;


	FlattenMatrix(A,  numberOfNonzerosPerRow);

	cl_int err;

	int plat_id = 0;
	int dev_id = 1;

	device = create_device(plat_id, dev_id);

	if(device == (cl_device_id)EXIT_SUCCESS) exit(1);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err < 0) {
		perror("Couldn't create a context");
		exit(1);
	}


	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_local_size), &max_local_size, NULL);

	//build kernels

	programSPMV = build_program(context, device, SPMV_PATH);
	programWAXPBY = build_program(context, device, WAXPBY_PATH);
	programProlongation = build_program(context, device, PROLONGATION_PATH);
	programDotProduct = build_program(context, device, DOTPRODUCT_PATH);
	programRestriction = build_program(context, device, RESTRICTION_PATH);
	programZerovec = build_program(context, device, ZEROVECTOR_PATH);

	//create kernels

	kernelSPMV = clCreateKernel(programSPMV, SPMV_FUNCTION, &err);
	if(err < 0) {
		perror("Couldn't create SPMV kernel");
		exit(1);
	};

	kernelWAXPBY = clCreateKernel(programWAXPBY, WAXPBY_FUNCTION, &err);
	if(err < 0) {
		perror("Couldn't create WAXPBY kernel");
		exit(1);
	};

	kernelProlongation = clCreateKernel(programProlongation, PROLONGATION_FUNCTION, &err);
	if(err < 0) {
		perror("Couldn't create Prolongation kernel");
		exit(1);
	};

	kernelDotProduct = clCreateKernel(programDotProduct, DOTPRODUCT_FUNCTION, &err);
	if(err < 0) {
		perror("Couldn't create Dot product kernel");
		exit(1);
	};

	kernelRestriction = clCreateKernel(programRestriction, RESTRICTION_FUNCTION, &err);
	if(err < 0) {
		perror("Couldn't create restriction kernel");
		exit(1);
	};

	kernelZerovec = clCreateKernel(programZerovec, ZEROVECTOR_FUNCTION, &err);
	if(err < 0) {
		perror("Couldn't create zerovector kernel");
		exit(1);
	};

	//create generic buffers

	f2cMatrix_buff = clCreateBuffer(context,
	CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
	sizeof(int)*A.localNumberOfRows, A.mgData->f2cOperator, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1);};


	//create buffers for SPMV

	spmv_output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR,
	sizeof(double)*data.Ap.localLength, NULL, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1); };

	spmv_nonzerosInRow_buff = clCreateBuffer(context,
	CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
	sizeof(char)*A.localNumberOfRows, A.nonzerosInRow, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1); };


	spmv_mtxIndL_buff = clCreateBuffer(context,
	CL_MEM_READ_ONLY,
	sizeof(int)*A.localNumberOfRows*numberOfNonzerosPerRow, NULL, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1);};

	spmv_matrixValues_buff = clCreateBuffer(context,
	CL_MEM_READ_ONLY,
	sizeof(double)*A.localNumberOfRows*numberOfNonzerosPerRow, NULL, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1);};

	spmv_xMatrix_buff = clCreateBuffer(context,
	CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
	sizeof(double)*data.p.localLength, NULL, &err);
	if(err < 0) { perror("Couldn't create spmv buffer"); exit(1);};


	//create buffers for WAXPBY


	waxpby_xMatrix_buff = clCreateBuffer(context,
	CL_MEM_READ_ONLY,
	sizeof(double)*x.localLength, NULL, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1); };

	waxpby_yMatrix_buff = clCreateBuffer(context,
	CL_MEM_READ_ONLY,
	sizeof(double)*x.localLength, NULL, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1);};

	waxpby_output_buffer = clCreateBuffer(context,
	CL_MEM_READ_WRITE,
	sizeof(double)*x.localLength, NULL, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1);};


	//create buffers for Dot Product

	dprod_xMatrix_buff = clCreateBuffer(context,
	CL_MEM_READ_ONLY,
	sizeof(double)*x.localLength, NULL, &err);
	if(err < 0) {
		perror("Couldn't create x buffer");
		exit(1);
	};
	dprod_yMatrix_buff = clCreateBuffer(context,
	CL_MEM_READ_WRITE,
	sizeof(double)*x.localLength, NULL, &err);
	if(err < 0) {
		perror("Couldn't create y buffer");
		exit(1);
	};


	dprod_temp_output = new double[max_local_size];

	for(int g = 0 ; g < (int)max_local_size ; g++){
		dprod_temp_output[g] = 0;
	}

	dprod_output_buff = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
	sizeof(double)*max_local_size, (void*)dprod_temp_output, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1); };



	//buffer for prolongation


	xfMatrix_buff = clCreateBuffer(context,
	CL_MEM_READ_WRITE,
	sizeof(double)*x.localLength, NULL, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1); };


	xcMatrix_buff = clCreateBuffer(context,
	CL_MEM_READ_ONLY,
	sizeof(double)*A.mgData->xc->localLength, NULL, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1);};


	//buffer for restriction

	restriction_axfMatrix_buff = clCreateBuffer(context,
	CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	sizeof(double)*A.mgData->Axf->localLength, A.mgData->Axf->values, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1); };

	//buffer for zero vector

	zerovec_xMatrix_buff = clCreateBuffer(context,
	CL_MEM_READ_WRITE,
	sizeof(double)*x.localLength, NULL, &err);
	if(err < 0) { perror("Couldn't create a buffer"); exit(1); };

	//set SPMV Kernel Params

	err = clSetKernelArg(kernelSPMV, 0, sizeof(unsigned int), (void *)&A.localNumberOfRows);
	err |= clSetKernelArg(kernelSPMV, 1, sizeof(int), (void *)&spmv_item_per_thread);
	err |= clSetKernelArg(kernelSPMV, 2, sizeof(cl_mem), &spmv_nonzerosInRow_buff);
	err |= clSetKernelArg(kernelSPMV, 3, sizeof(cl_mem), &spmv_mtxIndL_buff);
	err |= clSetKernelArg(kernelSPMV, 4, sizeof(cl_mem), &spmv_matrixValues_buff);
	err |= clSetKernelArg(kernelSPMV, 5, sizeof(cl_mem), &spmv_xMatrix_buff);
	err |= clSetKernelArg(kernelSPMV, 6, sizeof(cl_mem), &spmv_output_buffer);
	if(err < 0) {
		printf("Couldn't set an argument for the spmv kernel");
		exit(1);
	};


	//set WAXPBY Kernel Params

	int alpha, beta;

	err = clSetKernelArg(kernelWAXPBY, 0, sizeof(int), (void *)&A.localNumberOfRows);
	err |= clSetKernelArg(kernelWAXPBY, 1, sizeof(int), (void *)&waxpby_item_per_thread);
	err |= clSetKernelArg(kernelWAXPBY, 2, sizeof(double), &alpha);
	err |= clSetKernelArg(kernelWAXPBY, 3, sizeof(double), &beta);
	err |= clSetKernelArg(kernelWAXPBY, 4, sizeof(cl_mem), &waxpby_xMatrix_buff);
	err |= clSetKernelArg(kernelWAXPBY, 5, sizeof(cl_mem), &waxpby_yMatrix_buff);
	err |= clSetKernelArg(kernelWAXPBY, 6, sizeof(cl_mem), &waxpby_output_buffer);
	if(err < 0) {
		printf("Couldn't set an argument for WAXPBY the kernel");
		exit(1);
	};

	//set Prolongation Kernel Params

	err = clSetKernelArg(kernelProlongation, 0, sizeof(int), (void *)&A.mgData->rc->localLength);
	err |= clSetKernelArg(kernelProlongation, 1, sizeof(int), (void *)&prolongation_item_per_thread);
	err |= clSetKernelArg(kernelProlongation, 2, sizeof(cl_mem), &xfMatrix_buff);
	err |= clSetKernelArg(kernelProlongation, 3, sizeof(cl_mem), &xcMatrix_buff);
	err |= clSetKernelArg(kernelProlongation, 4, sizeof(cl_mem), &f2cMatrix_buff);
	if(err < 0) {
		printf("Couldn't set an argument for the Prolongation kernel");
		exit(1);
	};

	//set Restriction Kernel Params


	err = clSetKernelArg(kernelRestriction, 0, sizeof(int), (void *)&A.mgData->rc->localLength);
	err |= clSetKernelArg(kernelRestriction, 1, sizeof(int), (void *)&restriction_item_per_thread);
	err |= clSetKernelArg(kernelRestriction, 2, sizeof(cl_mem), &restriction_axfMatrix_buff);
	err |= clSetKernelArg(kernelRestriction, 3, sizeof(cl_mem), &xfMatrix_buff);
	err |= clSetKernelArg(kernelRestriction, 4, sizeof(cl_mem), &xcMatrix_buff);
	err |= clSetKernelArg(kernelRestriction, 5, sizeof(cl_mem), &f2cMatrix_buff);

	if(err < 0) {
		printf("Couldn't set an argument for the kernel");
		exit(1);
	};


	//set Dot Product Kernel Params

	err = clSetKernelArg(kernelDotProduct, 0, sizeof(int), (void *)&A.localNumberOfRows);
	err |= clSetKernelArg(kernelDotProduct, 1, sizeof(int), (void *)&dprod_item_per_thread);
	err |= clSetKernelArg(kernelDotProduct, 2, sizeof(cl_mem), &dprod_xMatrix_buff);
	err |= clSetKernelArg(kernelDotProduct, 3, sizeof(cl_mem), &dprod_yMatrix_buff);
	err |= clSetKernelArg(kernelDotProduct, 4, sizeof(cl_mem), &dprod_output_buff);
	err |= clSetKernelArg(kernelDotProduct, 5, max_local_size * 4* sizeof(double), (void *)NULL);
	if(err < 0) {
		printf("Couldn't set an argument for the dot product kernel");
		exit(1);
	};

	//set zero vector Kernel Params


	err = clSetKernelArg(kernelZerovec, 0, sizeof(int), (void *)&x.localLength);
	err |= clSetKernelArg(kernelZerovec, 1, sizeof(int), (void *)&zerovec_item_per_thread);
	err |= clSetKernelArg(kernelZerovec, 2, sizeof(cl_mem), &zerovec_xMatrix_buff);

	if(err < 0) {
		printf("Couldn't set an argument for the kernel");
		exit(1);
	};


	//create command queue
	
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	if(err < 0) {
		perror("Couldn't create a command queue");
		exit(1);
	};


	//---------------------------Enqueue Map Buffers for SMPV----------------------------------------

	/*
	data.p.values = (double*)clEnqueueMapBuffer(queue, spmv_xMatrix_buff,
												CL_TRUE, CL_MAP_READ, 0,
												sizeof(double)*data.p.localLength,
												0, NULL, NULL, &err );
	if(err < 0) {perror("Couldn't create map buffer");exit(1); };


	data.Ap.values = (double*)clEnqueueMapBuffer(queue, spmv_output_buffer,
												CL_TRUE, CL_MAP_WRITE, 0,
												sizeof(double)*data.p.localLength,
												0, NULL, NULL, &err );
	if(err < 0) {perror("Couldn't create map buffer");exit(1);};



	data.z.values = (double*)clEnqueueMapBuffer(queue, data_Z_values_buffer,
												CL_TRUE, CL_MAP_READ, 0,
												sizeof(double)*data.z.localLength,
												0, NULL, NULL, &err );
	if(err < 0) {perror("Couldn't create map buffer");exit(1);};
*/

	/*
	A.mgData->Axf->values = (double*)clEnqueueMapBuffer(queue, A_mgData_Axf_values_buffer,
												CL_TRUE, CL_MAP_WRITE, 0,
												sizeof(double)*A.mgData->Axf->localLength,
												0, NULL, NULL, &err );
	if(err < 0) {perror("Couldn't create map buffer");exit(1);};
*/

	//------------------------------Done Creating Kernels---------------------------------------


#ifdef HPCG_DEBUG
	if (rank==0) HPCG_fout << "Total problem setup time in main (sec) = " << mytimer() - t1 << endl;
#endif



	//////////////////////////////
	// Validation Testing Phase //
	//////////////////////////////



	if(rank == 0){
		printf("Start Validation Testing Phase");
	}


	#ifdef HPCG_DEBUG
	t1 = mytimer();
	#endif
	TestCGData testcg_data;
	testcg_data.count_pass = testcg_data.count_fail = 0;
	TestCG(A, data, b, x, testcg_data);

	TestSymmetryData testsymmetry_data;
	TestSymmetry(A, b, xexact, testsymmetry_data);

	#ifdef HPCG_DEBUG
	if (rank==0) HPCG_fout << "Total validation (TestCG and TestSymmetry) execution time in main (sec) = " << mytimer() - t1 << endl;
	#endif

	#ifdef HPCG_DEBUG
	t1 = mytimer();
	#endif

	///////////////////////////////////////
	// Reference SpMV+MG Timing Phase //
	///////////////////////////////////////

	// Call Reference SpMV and MG. Compute Optimization time as ratio of times in these routines



	if(rank == 0){
		printf("\nStart SPMV+MG Timing Phase");
	}


	local_int_t nrow = A.localNumberOfRows;
	local_int_t ncol = A.localNumberOfColumns;

	Vector x_overlap, b_computed;
	InitializeVector(x_overlap, ncol); // Overlapped copy of x vector
	InitializeVector(b_computed, nrow); // Computed RHS vector


	// Record execution time of reference SpMV and MG kernels for reporting times
	// First load vector with random values
	FillRandomVector(x_overlap);

	int numberOfCalls = 10;
	double t_begin = mytimer();
	for (int i=0; i< numberOfCalls; ++i) {
		ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
		if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
		ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap
		if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
	}
	times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.
	#ifdef HPCG_DEBUG
	if (rank==0) HPCG_fout << "Total SpMV+MG timing phase execution time in main (sec) = " << mytimer() - t1 << endl;
	#endif

	///////////////////////////////
	// Reference CG Timing Phase //
	///////////////////////////////


	if(rank == 0){
		printf("\nStart Refrence CG Timing Phase");
	}


	#ifdef HPCG_DEBUG
	t1 = mytimer();
	#endif
	int global_failure = 0; // assume all is well: no failures

	int niters = 0;
	int totalNiters_ref = 0;
	double normr = 0.0;
	double normr0 = 0.0;
	int refMaxIters = 50;
	numberOfCalls = 1; // Only need to run the residual reduction analysis once

	// Compute the residual reduction for the natural ordering and reference kernels
	std::vector< double > ref_times(9,0.0);
	double tolerance = 0.0; // Set tolerance to zero to make all runs do maxIters iterations
	int err_count = 0;
	for (int i=0; i< numberOfCalls; ++i) {
		ZeroVector(x);
		ierr = CG_ref( A, data, b, x, refMaxIters, tolerance, niters, normr, normr0, &ref_times[0], true);
		if (ierr) ++err_count; // count the number of errors in CG
		totalNiters_ref += niters;
	}
	if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to reference CG." << endl;
	double refTolerance = normr / normr0;

	//////////////////////////////
	// Optimized CG Setup Phase //
	//////////////////////////////



	if(rank == 0){
		printf("\nStart Optimized CG Setup Phase");
	}

	niters = 0;
	normr = 0.0;
	normr0 = 0.0;
	err_count = 0;
	int tolerance_failures = 0;

	int optMaxIters = 10*refMaxIters;
	int optNiters = 0;
	double opt_worst_time = 0.0;

	std::vector< double > opt_times(9,0.0);

	// Compute the residual reduction and residual count for the user ordering and optimized kernels.
	for (int i=0; i< numberOfCalls; ++i) {
		ZeroVector(x); // start x at all zeros
		double last_cummulative_time = opt_times[0];
		ierr = CG( A, data, b, x, optMaxIters, refTolerance, niters, normr, normr0, &opt_times[0], true);
		if (ierr) ++err_count; // count the number of errors in CG
		if (normr / normr0 > refTolerance) ++tolerance_failures; // the number of failures to reduce residual

		// pick the largest number of iterations to guarantee convergence
		if (niters > optNiters) optNiters = niters;

		double current_time = opt_times[0] - last_cummulative_time;
		if (current_time > opt_worst_time) opt_worst_time = current_time;
	}

	#ifndef HPCG_NOMPI


	printf("\nMPI Enabled");


	// Get the absolute worst time across all MPI ranks (time in CG can be different)
	double local_opt_worst_time = opt_worst_time;
	MPI_Allreduce(&local_opt_worst_time, &opt_worst_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	#endif


	if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to optimized CG." << endl;
	if (tolerance_failures) {
		global_failure = 1;
		if (rank == 0)
		HPCG_fout << "Failed to reduce the residual " << tolerance_failures << " times." << endl;
	}

	///////////////////////////////
	// Optimized CG Timing Phase //
	///////////////////////////////

	// Here we finally run the benchmark phase
	// The variable total_runtime is the target benchmark execution time in seconds

	if (rank==0) {
		char c;
		std::cout << "Press key to continue"<< std::endl;
		std::cin.get(c);
	}

	if(rank == 0){
		printf("Start CG Optimized timing phase - Running Benchmark");
	}

	double total_runtime = params.runningTime;
	int numberOfCgSets = int(total_runtime / opt_worst_time) + 1; // Run at least once, account for rounding

	#ifdef HPCG_DEBUG
	if (rank==0) {
		HPCG_fout << "Projected running time: " << total_runtime << " seconds" << endl;
		HPCG_fout << "Number of CG sets: " << numberOfCgSets << endl;
	}
	#endif

	/* This is the timed run for a specified amount of time. */

	optMaxIters = optNiters;
	double optTolerance = 0.0;  // Force optMaxIters iterations
	TestNormsData testnorms_data;
	testnorms_data.samples = numberOfCgSets;
	testnorms_data.values = new double[numberOfCgSets];

	if(rank == 0){
		printf("\nEntering CG loop with Total CG Sets: %d",numberOfCgSets);
	}



	for (int i=0; i< numberOfCgSets; ++i) {
		ZeroVector(x); // Zero out x
		ierr = CG( A, data, b, x, optMaxIters, optTolerance, niters, normr, normr0, &times[0], true);
		if (ierr) HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
		if (rank==0) HPCG_fout << "Call [" << i << "] Scaled Residual [" << normr/normr0 << "]" << " normr = " << normr <<" normr0 "<< normr0 <<endl;
		testnorms_data.values[i] = normr/normr0; // Record scaled residual from this run
	}

	#ifdef HPCG_DETAILED_DEBUG
	if (geom->size == 1) WriteProblem(*geom, A, b, x, xexact);
	#endif



	if(rank == 0){
		printf("\nExiting CG loop");
	}



	// Compute difference between known exact solution and computed solution
	// All processors are needed here.
	#ifdef HPCG_DEBUG
	double residual = 0;
	ierr = ComputeResidual(A.localNumberOfRows, x, xexact, residual);
	if (ierr) HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n" << endl;
	if (rank==0) HPCG_fout << "Difference between computed and exact  = " << residual << ".\n" << endl;
	#endif



	printf("\nTest Results");



	// Test Norm Results
	ierr = TestNorms(testnorms_data);
	if (ierr) HPCG_fout << "Error in testing: " << ierr << ".\n" << endl;

	////////////////////
	// Report Results //
	////////////////////

	// Report results to YAML file
	ReportResults(A, numberOfMgLevels, numberOfCgSets, refMaxIters, optMaxIters, &times[0], testcg_data, testsymmetry_data, testnorms_data, global_failure);

	// Clean up
	DeleteMatrix(A); // This delete will recursively delete all coarse grid data
	DeleteCGData(data);
	DeleteVector(x);
	DeleteVector(b);
	DeleteVector(xexact);
	DeleteVector(x_overlap);
	DeleteVector(b_computed);
	delete [] testnorms_data.values;


	/*

char c;
std::cout << "Press key to continue"<< std::endl;
std::cin.get(c);
printf("Running SPMV");

Vector & r = data.r; // Residual vector
Vector & z = data.z; // Preconditioned residual vector
Vector & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
Vector & Ap = data.Ap;

FlattenMatrix(A,  numberOfNonzerosPerRow);
runSPMV(A.localNumberOfRows,
		numberOfNonzerosPerRow,
		A.nonzerosInRow,
		A.flat_mtxIndL,
		A.flat_matrixValues,
		p.values,
		p.localLength,
		Ap.values,
		Ap.localLength,
			device,
			context,
			kernelSPMV);

printf("\nCompleted SPMV. Running WAXPBY\n");
std::cout << "Press key to continue"<< std::endl;
std::cin.get(c);

runWAXPBY(A.localNumberOfRows,1.0,-1.0,
		b.values,
		b.localLength,
		Ap.values,
		Ap.localLength,
		r.values,
		r.localLength,
		device,
		context,
		kernelWAXPBY);



printf("\nCompleted WAXPBY. Running Prolongation\n");


double * xfv = x.values;
double * xcv = A.mgData->xc->values;
int * f2c = A.mgData->f2cOperator;
int nc = A.mgData->rc->localLength;



printf("nc %d x %d xc %d fc %d\n",nc,x.localLength,A.mgData->xc->localLength,A.localNumberOfRows);



std::cout << "Press key to continue"<< std::endl;
std::cin.get(c);




runProlongation(0, 1,nc,xfv,x.localLength,
		xcv,
		A.mgData->xc->localLength,
		f2c,
		A.localNumberOfRows,
			device,
			context,
			kernelProlongation);



printf("\nCompleted Prolongation. Running Dot Product\n");

std::cout << "Press key to continue"<< std::endl;
std::cin.get(c);


double * rfv = r.values;
double * bfv = b.values;


runDotProduct(0, 1,A.localNumberOfRows,rfv,r.localLength,bfv,b.localLength,xfv,x.localLength);

printf("\nCompleted Dot Product. Running Restriction\n");

double * Axfv = A.mgData->Axf->values;
double * rcv = A.mgData->rc->values;

runRestriction(0, 1,nc,Axfv,
		A.mgData->Axf->localLength,rfv,
		r.localLength,rcv,
		A.mgData->rc->localLength,f2c,
		A.localNumberOfRows,
			device,
			context,
			kernelRestriction);


printf("\nCompleted Restriction.");

double *output_vec = new double[4096];

double a_vec[4096], b_vec[4096];

for(int i=0; i<4096; i++) {
	a_vec[i] = 1;
	b_vec[i] = 1;
	}

double res = runDotProduct(A.localNumberOfRows,a_vec,4096,b_vec,b.localLength,output_vec,4096, device,
			context,
			kernelDotProduct);

printf("dot prod res :%f\n", res);

*/



	HPCG_Finalize();


	//cleanup

	clReleaseKernel(kernelSPMV);
	clReleaseKernel(kernelWAXPBY);
	clReleaseKernel(kernelProlongation);
	clReleaseKernel(kernelDotProduct);
	clReleaseKernel(kernelRestriction);
	clReleaseProgram(programSPMV);
	clReleaseProgram(programWAXPBY);
	clReleaseProgram(programProlongation);
	clReleaseProgram(programDotProduct);
	clReleaseProgram(programRestriction);
	clReleaseContext(context);
	clReleaseMemObject(spmv_output_buffer);
	clReleaseMemObject(spmv_nonzerosInRow_buff);
	clReleaseMemObject(spmv_mtxIndL_buff);
	clReleaseMemObject(spmv_matrixValues_buff);
	clReleaseMemObject(spmv_xMatrix_buff);
	clReleaseMemObject(waxpby_xMatrix_buff);
	clReleaseMemObject(waxpby_yMatrix_buff);
	clReleaseMemObject(waxpby_output_buffer);
	clReleaseMemObject(dprod_xMatrix_buff);
	clReleaseMemObject(dprod_yMatrix_buff);
	clReleaseMemObject(dprod_output_buff);
	clReleaseMemObject(xfMatrix_buff);
	clReleaseMemObject(xcMatrix_buff);
	clReleaseMemObject(f2cMatrix_buff);
	clReleaseMemObject(restriction_axfMatrix_buff);
	clReleaseMemObject(zerovec_xMatrix_buff);
	clReleaseCommandQueue(queue);


	// Finish up
#ifndef HPCG_NOMPI
	MPI_Finalize();
#endif
	return 0 ;
}






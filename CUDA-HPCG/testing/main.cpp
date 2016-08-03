
#ifndef HPCG_NOMPI
#include <mpi.h> // If this routine is not compiled with HPCG_NOMPI
#endif

#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>

#include "hpcg.hpp"

#include <cuda.h>


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
#include "../CUDA_src/cuSPMV.h"
#include "../CUDA_src/cuGetDevice.h"
#include "../CUDA_src/cuDProd.h"
#include "../CUDA_src/cuWAXPBY.h"
#include "../CUDA_src/cuProlongation.h"
#include "../CUDA_src/cuRestriction.h"
#define HPCG_DEBUG
#define HPCG_DETAILED_DEBUG




//declare CUDA objects here
char * spmv_AnonzerosInRow_d;
int * spmv_AmtxIndL_d;
double * spmv_AmatrixValues_d;
double * yMatrix_d;

int spmv_item_per_thread,dprod_item_per_thread, dprod_block_size, dprodNumBlocks;
int spmv_block_size,spmvNumBlocks;

double *spmv_output_buffer, *spmv_matrixValues_buff,*spmv_xMatrix_buff;
double *dprod_output_buff, *dprod_yMatrix_buff,*dprod_xMatrix_buff;
double *dprod_output;

char *spmv_nonzerosInRow_buff;
int *spmv_mtxIndL_buff;

int waxpby_block_size, waxpby_item_per_thread,waxpbyNumBlocks;
double *waxpby_xMatrix_buff, *waxpby_yMatrix_buff, *waxpby_wMatrix_buff;


int prolongation_block_size,prolongationNumBlocks,prolongation_item_per_thread;
int * prolongation_f2cMatrix_buff;
double * prolongation_xfMatrix_buff, *prolongation_xcMatrix_buff;



int restriction_block_size,restrictionNumBlocks,restriction_item_per_thread;
double * restriction_axfMatrix_d_buff;




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




	//------------------------------initialize CUDA Kernels---------------------------


	cuGetDevice();
	local_int_t numberOfNonzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil

	int dprodMinGridSize=0;
	int waxpbyMinGridSize=0;
	int spmvMinGridSize=0;
	int prolongationMinGridSize =0;
	int restrictionMinGridSize =0;

	spmv_item_per_thread = 1;
	dprod_item_per_thread = 1;
	waxpby_item_per_thread = 1;
	prolongation_item_per_thread = 1;
	restriction_item_per_thread =1;

	FlattenMatrix(A,  numberOfNonzerosPerRow);

	//spmv buffers-------------------------------------------------------------------------------
	cudaMalloc((void **) &spmv_output_buffer, sizeof(double)*data.Ap.localLength);
	cudaMalloc((void **) &spmv_nonzerosInRow_buff, sizeof(char)*A.localNumberOfRows);
	cudaMalloc((void **) &spmv_mtxIndL_buff, sizeof(int)*A.localNumberOfRows*numberOfNonzerosPerRow);
	cudaMalloc((void **) &spmv_matrixValues_buff, sizeof(double)*A.localNumberOfRows*numberOfNonzerosPerRow);
	cudaMalloc((void **) &spmv_xMatrix_buff, sizeof(double)*data.p.localLength);




	cudaMemcpy( spmv_nonzerosInRow_buff, A.nonzerosInRow, sizeof(char)*A.localNumberOfRows, cudaMemcpyHostToDevice );
	cudaMemcpy( spmv_mtxIndL_buff, A.flat_mtxIndL, sizeof(int)*A.localNumberOfRows*numberOfNonzerosPerRow, cudaMemcpyHostToDevice );
	cudaMemcpy( spmv_matrixValues_buff, A.flat_matrixValues, sizeof(double)*A.localNumberOfRows*numberOfNonzerosPerRow, cudaMemcpyHostToDevice );


	spmv_block_size = getSPMVBlockSize(A.localNumberOfRows/spmv_item_per_thread,spmvMinGridSize,spmv_block_size);
	spmvNumBlocks = (A.localNumberOfRows + spmv_block_size - 1) / spmv_block_size;
	//dprod buffers--------------------------------------------------------------------------------


	dprod_block_size = getDProdBlockSize(x.localLength/dprod_item_per_thread,dprodMinGridSize, dprod_block_size);

	printf("block size %d\n",dprod_block_size);

	cudaMalloc((void **) &dprod_output_buff, sizeof(double)*dprod_block_size);
	cudaMalloc((void **) &dprod_yMatrix_buff, sizeof(double)*x.localLength);
	cudaMalloc((void **) &dprod_xMatrix_buff, sizeof(double)*x.localLength);


	dprodNumBlocks = (x.localLength + dprod_block_size - 1) / dprod_block_size;

	dprod_output = new double[dprodNumBlocks];

	for(int g = 0 ; g < (int)dprodNumBlocks ; g++){
		dprod_output[g] = 0;
	}

	cudaMemcpy( dprod_output_buff, dprod_output, sizeof(double)*dprodNumBlocks, cudaMemcpyHostToDevice );



	//waxpby buffers--------------------------------------------------------------------------------


	waxpby_block_size = getDProdBlockSize(x.localLength/waxpby_item_per_thread,waxpbyMinGridSize, waxpby_block_size);
	waxpbyNumBlocks = (x.localLength + waxpby_block_size - 1) / waxpby_block_size;

	cudaMalloc((void **) &waxpby_xMatrix_buff, sizeof(double)*x.localLength);
	cudaMalloc((void **) &waxpby_yMatrix_buff, sizeof(double)*x.localLength);
	cudaMalloc((void **) &waxpby_wMatrix_buff, sizeof(double)*x.localLength);

	//prolongation buffers--------------------------------------------------------------------------------
	
	prolongation_block_size = getProlongationBlockSize(x.localLength/prolongation_item_per_thread,prolongationMinGridSize, prolongation_block_size);
	prolongationNumBlocks = (x.localLength + prolongation_block_size - 1) / prolongation_block_size;

	cudaMalloc((void **) &prolongation_xfMatrix_buff, sizeof(double)*x.localLength);
	cudaMalloc((void **) &prolongation_xcMatrix_buff, sizeof(double)*A.mgData->xc->localLength);
	cudaMalloc((void **) &prolongation_f2cMatrix_buff, sizeof(double)*A.localNumberOfRows);

	//cudaMemcpy(prolongation_xcMatrix_buff , A.mgData->xc->values, sizeof(double)*A.mgData->xc->localLength, cudaMemcpyHostToDevice );
	cudaMemcpy(prolongation_f2cMatrix_buff, A.mgData->f2cOperator, sizeof(int)*A.localNumberOfRows, cudaMemcpyHostToDevice );
	
	//restriction buffers--------------------------------------------------------------------------------

	restriction_block_size = getRestrictionBlockSize(A.mgData->rc->localLength/restriction_item_per_thread,restrictionMinGridSize, restriction_block_size);
	restrictionNumBlocks = (x.localLength + restriction_block_size - 1) / restriction_block_size;
	
	cudaMalloc((void **) &restriction_axfMatrix_d_buff, sizeof(double)*A.mgData->Axf->localLength);

	cudaMemcpy(restriction_axfMatrix_d_buff , A.mgData->Axf->values, sizeof(double)*A.mgData->xc->localLength, cudaMemcpyHostToDevice );
	

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



	HPCG_Finalize();


	cudaFree(spmv_output_buffer);
	cudaFree(spmv_matrixValues_buff);
	cudaFree(spmv_xMatrix_buff);
	cudaFree(spmv_nonzerosInRow_buff);
	cudaFree(spmv_mtxIndL_buff);



	// Finish up
#ifndef HPCG_NOMPI
	MPI_Finalize();
#endif
	return 0 ;
}






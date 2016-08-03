
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include "../OCL_src/oclSPMV.h"
#include "ExchangeHalo.hpp"

extern cl_kernel kernelSPMV;
extern int spmv_item_per_thread;
extern cl_command_queue queue;
extern cl_mem spmv_output_buffer,spmv_xMatrix_buff;


/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

  A.isSpmvOptimized = true;
	#ifndef HPCG_NOMPI
		ExchangeHalo(A,x);
	#endif

  //ComputeSPMV_ref(A, x, y)
  return(runSPMV(A.localNumberOfRows,
	  		x.values,
	  		x.localLength,
	  		y.values,
	  		y.localLength,
			kernelSPMV,
			spmv_output_buffer,
			spmv_xMatrix_buff,
	  		queue,
			spmv_item_per_thread));

}

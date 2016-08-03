
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
#include "../CUDA_src/cuSPMV.h"



extern int spmv_item_per_thread;


extern double * spmv_output_buffer, * spmv_matrixValues_buff, *spmv_xMatrix_buff  ;
extern char *spmv_nonzerosInRow_buff;
extern int *spmv_mtxIndL_buff;
extern int spmv_block_size,spmvNumBlocks;


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


  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  A.isSpmvOptimized = true;

  return(runSPMV(A.localNumberOfRows, 27,
  		  x.values, x.localLength, y.values, y.localLength, spmv_nonzerosInRow_buff,
		spmv_mtxIndL_buff, spmv_matrixValues_buff, spmv_xMatrix_buff,spmv_output_buffer,
		spmv_item_per_thread,spmv_block_size,spmvNumBlocks));

//return(ComputeSPMV_ref(A, x, y));

}

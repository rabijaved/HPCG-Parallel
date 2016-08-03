
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "ComputeSPMV.hpp"
#include "../CUDA_src/cuProlongation.h"
#include <stdio.h>
#include <string.h>
#include "../CUDA_src/cuRestriction.h"
extern int prolongation_block_size,prolongationNumBlocks,prolongation_item_per_thread;
extern int * prolongation_f2cMatrix_buff;
extern double * prolongation_xfMatrix_buff, *prolongation_xcMatrix_buff;

extern int restriction_block_size,restrictionNumBlocks,restriction_item_per_thread;
extern double * restriction_axfMatrix_d_buff;


/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  A.isMgOptimized = true;
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  //ZeroVector(x); // initialize x to zero

  memset (x.values,0,sizeof(double)*x.localLength);

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return(ierr);
    
	ComputeSPMV(A, x, *A.mgData->Axf);
    //ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf); if (ierr!=0) return(ierr);
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_ref(A, r);  if (ierr!=0) return(ierr);


  runRestriction(A.mgData->rc->localLength,
      r.values, r.localLength, A.mgData->rc->values, A.mgData->rc->localLength,
      A.mgData->Axf->values, A.mgData->Axf->localLength,
      restriction_axfMatrix_d_buff,
      prolongation_xfMatrix_buff,
      prolongation_xcMatrix_buff,
      prolongation_f2cMatrix_buff,
      restriction_item_per_thread, restriction_block_size, restrictionNumBlocks);



    //ierr = ComputeMG_ref(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return(ierr);

    
	runProlongation(x.localLength,
		x.values, x.localLength,
    A.mgData->xc->values, A.mgData->xc->localLength,
		prolongation_xfMatrix_buff, prolongation_xcMatrix_buff, prolongation_f2cMatrix_buff,
	    prolongation_item_per_thread, prolongation_block_size,prolongationNumBlocks);

    //ierr = ComputeProlongation_ref(A, x);  if (ierr!=0) return(ierr);
    

    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return(ierr);
  }
  else {
    ierr = ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return(ierr);
  }
  return(0);


}


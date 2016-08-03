
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

#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include "../OCL_src/oclSPMV.h"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#include "../OCL_src/oclProlongation.h"
#include "../OCL_src/oclRestriction.h"
#include "../OCL_src/oclZeroVector.h"
#include <stdio.h>
#include <string.h>

extern cl_mem xfMatrix_buff, xcMatrix_buff,f2cMatrix_buff;
extern cl_mem zerovec_xMatrix_buff,restriction_axfMatrix_buff;
extern int zerovec_item_per_thread,prolongation_item_per_thread, restriction_item_per_thread;
extern cl_command_queue queue;
extern cl_kernel kernelZerovec,kernelProlongation, kernelRestriction;


/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

	  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values


	  /*runZeroVector(x.localLength,
	  		x.values, kernelZerovec, zerovec_xMatrix_buff,
			queue, zerovec_item_per_thread);
	  */
	  memset (x.values,0,sizeof(double)*x.localLength);
	  //ZeroVector(x); // initialize x to zero

	  int ierr = 0;
	  if (A.mgData!=0) { // Go to next coarse level if defined
	    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
	    for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
	    if (ierr!=0) return(ierr);


	    //ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf); if (ierr!=0) return(ierr);

	    ierr = ComputeSPMV(A, x, *A.mgData->Axf);
	    if (ierr!=0) return(ierr);

	// Perform restriction operation using simple injection


	    ierr = runRestriction(A.mgData->rc->localLength,
	    		A.mgData->Axf->values, A.mgData->Axf->localLength,
				r.values, r.localLength, A.mgData->rc->values, A.mgData->rc->localLength,
				A.mgData->f2cOperator, A.localNumberOfRows, kernelRestriction,
				restriction_axfMatrix_buff, xfMatrix_buff ,xcMatrix_buff, f2cMatrix_buff,
				queue, restriction_item_per_thread);
	    if (ierr!=0) return(ierr);

	    //ierr = ComputeRestriction_ref(A, r);  if (ierr!=0) return(ierr);

	    ierr = ComputeMG_ref(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return(ierr);

	    ierr = runProlongation(A.mgData->rc->localLength,x.values,x.localLength, A.mgData->xc->values, A.mgData->xc->localLength,
	    	A.mgData->f2cOperator, A.localNumberOfRows,kernelProlongation,
			xfMatrix_buff,xcMatrix_buff,
			f2cMatrix_buff, queue,
			prolongation_item_per_thread);
	    if (ierr!=0) return(ierr);



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


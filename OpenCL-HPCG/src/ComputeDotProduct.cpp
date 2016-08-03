
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"
#include "../OCL_src/oclDProduct.h"



#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

extern cl_mem dprod_xMatrix_buff, dprod_yMatrix_buff, dprod_output_buff;
extern cl_kernel kernelDotProduct;
extern double *dprod_temp_output;
extern int dprod_item_per_thread;
extern size_t max_local_size;
extern cl_command_queue queue;


/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
  isOptimized = true;
  double temp;



  temp = runDotProduct(n, x.values,x.localLength,y.values,y.localLength,dprod_temp_output,(int)max_local_size,
			kernelDotProduct,dprod_xMatrix_buff, dprod_yMatrix_buff, dprod_output_buff,
			queue,dprod_item_per_thread);
 //int res = ComputeDotProduct_ref(n, x, y, result, time_allreduce);

result = temp;


  return(0);
}

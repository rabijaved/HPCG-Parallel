#ifndef BUILDKERNEL_H
#define BUILDKERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);

#ifdef __cplusplus
}
#endif

#endif


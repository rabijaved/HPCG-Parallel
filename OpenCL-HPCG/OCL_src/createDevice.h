#ifndef CREATEDEVICE_H
#define CREATEDEVICE_H

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


cl_device_id create_device(int plat_id, int dev_id);


#ifdef __cplusplus
}
#endif

#endif


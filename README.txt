--------------------Acknowledgements-----------------------

The HPCG code was adopted from the official HPCG source website
and can be accessed from https://software.sandia.gov/hpcg/download.php

Please refer to List-of-Files-and-Description.pdf, for the list of files
that were created and adopted from HPCG for modifications.  

A complete list of references is avilable in the project report document

-----------------------------------------------------------

Two versions of the code have been included in the respective folders.
The instructions and requirements for executing the OpenCL and CUDA 
versions of HPCG are mentioned below. For a description of the files, refer to 
List of Files and Description.pdf

The Directory structure for both versions is similar:

src : HPCG source files
build2: Makefile for project 
build2/bin: Contains the compiled binaries
build2/bin/xhpcg: Compiled executable
build2/bin/hpcg.dat: File specifying problem dimensions and run time

OCL_src (OpenCL only): OpenCL source files and kernels
CUDA_src (CUDA only): CUDA source files and kernels

-------------------------CUDA-HPCG------------------------

How to run the code:

Navigate to \CUDA-HPCG\build2
Edit Makefile to include the correct CUDA library directory
Check PATH system variable to include CUDA bin directory
Check LD_LIBRARY_PATH system variable to include CUDA lib directory
Execute terminal command :make clean
Execute terminal command :make

Navigate to \CUDA-HPCG\build2\bin

Edit hcpg.dat to spcify the run time and problem size
Run the benchmark : ./xhpcg

------------------------OpenCL-HPCG-----------------------

How to run the code:

Navigate to \OpenCl-HPCG\build2
Edit Makefile to include the correct OpenCl library directory
Check PATH system variable to include OpenCl bin directory
Check LD_LIBRARY_PATH system variable to include OpenCl lib directory
Execute terminal command :make clean
Execute terminal command :make

Navigate to \OpenCl-HPCG\build2\bin

Edit hcpg.dat to spcify the run time and problem size
Run the benchmark : ./xhpcg

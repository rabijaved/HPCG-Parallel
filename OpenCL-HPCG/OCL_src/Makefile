PROJ1=oclProlongation
PROJ2=oclDProduct
PROJ3=oclRestriction
PROJ4=oclSPMV
PROJ5=oclWAXPBY
PROJ6=createDevice
PROJ7=buildKernel
PROJ8=oclZeroVector

BUILD=build2

CC=gcc

CFLAGS = -std=c99 -w -DUNIX -g -DDEBUG


# Check for 32-bit vs 64-bit
PROC_TYPE = $(strip $(shell uname -m | grep 64))
 
# Check for Mac OS
OS = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN = $(strip $(findstring DARWIN, $(OS)))



# MacOS System
ifneq ($(DARWIN),)
	CFLAGS += -DMAC
	LIBS=-framework OpenCL -lrt

	ifeq ($(PROC_TYPE),)
		CFLAGS+=-arch i386
	else
		CFLAGS+=-arch x86_64
	endif
else

# Linux OS
LIBS=-lOpenCL -lrt -lm
ifeq ($(PROC_TYPE),)
	CFLAGS+=-m32
else
	CFLAGS+=-m64
endif

# Check for Linux-AMD
ifdef AMDAPPSDKROOT
   INC_DIRS=. $(AMDAPPSDKROOT)/include
	ifeq ($(PROC_TYPE),)
		LIB_DIRS=$(AMDAPPSDKROOT)/lib/x86
	else
		LIB_DIRS=$(AMDAPPSDKROOT)/lib/x86_64
	endif
else

# Check for Linux-Nvidia
ifdef NVSDKCOMPUTE_ROOT
   INC_DIRS=. $(NVSDKCOMPUTE_ROOT)/OpenCL/common/inc
endif

endif
endif

all: $(PROJ1) $(PROJ2) $(PROJ3) $(PROJ4) $(PROJ5) $(PROJ6) $(PROJ7) $(PROJ8) copy

clean: 
	rm src/*.o
	rm ../$(BUILD)/src/$(PROJ1).o
	rm ../$(BUILD)/src/$(PROJ2).o
	rm ../$(BUILD)/src/$(PROJ3).o
	rm ../$(BUILD)/src/$(PROJ4).o
	rm ../$(BUILD)/src/$(PROJ5).o
	rm ../$(BUILD)/src/$(PROJ6).o
	rm ../$(BUILD)/src/$(PROJ7).o
	rm ../$(BUILD)/src/$(PROJ8).o
	rm ../$(BUILD)/src/*.cl

copy: 
	cp src/*.o ../$(BUILD)/src/
	cp *.cl ../$(BUILD)/src/

$(PROJ1): $(PROJ1).c
	$(CC) -c -o src/$(PROJ1).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ2): $(PROJ2).c
	$(CC) -c -o src/$(PROJ2).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ3): $(PROJ3).c
	$(CC) -c -o src/$(PROJ3).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ4): $(PROJ4).c
	$(CC) -c -o src/$(PROJ4).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ5): $(PROJ5).c
	$(CC) -c -o src/$(PROJ5).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ6): $(PROJ6).c
	$(CC) -c -o src/$(PROJ6).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ7): $(PROJ7).c
	$(CC) -c -o src/$(PROJ7).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ8): $(PROJ8).c
	$(CC) -c -o src/$(PROJ8).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

.PHONY: clean

PROJ1=cuSPMV
PROJ2=cuGetDevice
PROJ3=cuDProd
PROJ4=cuWAXPBY
PROJ5=cuProlongation
PROJ6=cuRestriction
BUILD=build2

CC=nvcc


all: $(PROJ1) $(PROJ2) $(PROJ3) $(PROJ4) $(PROJ5) $(PROJ6) copy

clean: 
	rm src/*.o
	rm ../$(BUILD)/src/$(PROJ1).o
	rm ../$(BUILD)/src/$(PROJ2).o
	rm ../$(BUILD)/src/$(PROJ3).o
	rm ../$(BUILD)/src/$(PROJ4).o
	rm ../$(BUILD)/src/$(PROJ5).o
	rm ../$(BUILD)/src/$(PROJ6).o
copy: 
	cp src/*.o ../$(BUILD)/src/

$(PROJ1): $(PROJ1).cu
	$(CC) -c -o src/$(PROJ1).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ2): $(PROJ2).cu
	$(CC) -c -o src/$(PROJ2).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ3): $(PROJ3).cu
	$(CC) -c -o src/$(PROJ3).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ4): $(PROJ4).cu
	$(CC) -c -o src/$(PROJ4).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)


$(PROJ5): $(PROJ5).cu
	$(CC) -c -o src/$(PROJ5).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

$(PROJ6): $(PROJ6).cu
	$(CC) -c -o src/$(PROJ6).o $(CFLAGS) $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

.PHONY: clean

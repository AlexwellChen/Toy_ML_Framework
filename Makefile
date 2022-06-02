IDIR=include
CBLASDIR=/opt/homebrew/Cellar/openblas/0.3.17/include
CBLASLIB = /opt/homebrew/Cellar/openblas/0.3.17/lib
CXX=g++-11
CXXFLAGS=-I$(IDIR) -I$(CBLASDIR) -L$(CBLASLIB) -std=c++11 -fopenmp -O3

ODIR=src
LDIR =../lib

LIBS=-lm -fopenmp -lblas

_DEPS = deep_core.h vector_ops.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = deep_core.o vector_ops.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< ${CXXFLAGS}

%.o: %.cxx $(DEPS)
	$(CXX) -c -o $@ $< ${CXXFLAGS}

nnetwork_mpi.o:
	$(MPICXX) -c -o $@ nnetwork.cxx ${CXXFLAGS} -DUSE_MPI

nnetwork.o:
	$(CXX) -c -o $@ nnetwork.cxx ${CXXFLAGS} 

nnetwork_mpi: $(OBJ) nnetwork_mpi.o
	$(CXX) -o $@ $^ $(LIBS)

nnetwork: $(OBJ) nnetwork.o
	$(CXX) -o $@ $^ $(LIBS)

run_pthreads:
	./nnetwork

run_parallel:
	mpirun -np 4 ./nnetwork_mpi

all: clean nnetwork_mpi nnetwork
.PHONY: clean

default: clean nnetwork 

.DEFAULT_GOAL := default

clean:
	rm -f $(ODIR)/*.o *.o nnetwork_mpi nnetwork

CC = g++

CFLAGS = -O3 -std=c++11 -DNDEBUG -I utils/ -Wall -funroll-loops
LDFLAGS = -pthread -lm

EIGEN_CFLAGS = -DEIGEN_NO_DEBUG -DEIGEN_USE_MKL_ALL \
			   -I utils/eigen -I /opt/intel/mkl/include/

OMP = 1
MKL = 1

ifdef MKL
	MKL_LDFLAGS = -L/opt/intel/mkl/lib/intel64/ \
				  -lmkl_rt -lmkl_gnu_thread -lmkl_core
endif

ifdef OMP
	OMP_CFLAGS = -fopenmp
	OMP_LDFLAGS = -fopenmp
endif

ALL_CFLAGS = $(OMP_CFLAGS) $(CFLAGS) $(EIGEN_CFLAGS)
ALL_LDFLAGS = $(OMP_LDFLAGS) $(MKL_LDFLAGS) $(LDFLAGS)

OBJS = Config.o Dataset.o DependencySent.o DependencyTree.o \
	   Configuration.o ParsingSystem.o ArcStandard.o \

OBJS_MAT = $(OBJS) Classifier.o DependencyParser.o
OBJS_EIGEN = $(OBJS) ClassifierEigen.o DependencyParserEigen.o

HEADERS = Config.h Dataset.h DependencySent.h DependencyTree.h \
		  Configuration.h ParsingSystem.h ArcStandard.h Util.h \

HEADERS_MAT = Classifier.h DependencyParser.h
HEADERS_EIGEN = ClassifierEigen.h DependencyParserEigen.h

all: nndep proj nndep_eigen eval

eval : eval.o $(OBJS) $(HEADERS)
	$(CC) -o eval eval.o $(OBJS) $(CFLAGS)

proj : proj.o $(OBJS) $(HEADERS)
	$(CC) -o proj proj.o $(OBJS) $(CFLAGS)

nndep : nndep.o $(OBJS_MAT) $(HEADERS) $(HEADERS_MAT)
	$(CC) -o nndep nndep.o $(OBJS_MAT) $(CFLAGS) $(OMP_CFLAGS) $(LDFLAGS)

nndep_eigen : nndep_eigen.o $(OBJS_EIGEN) $(HEADERS) $(HEADERS_EIGEN)
	$(CC) -o nndep_eigen nndep_eigen.o $(OBJS_EIGEN) $(ALL_CFLAGS) $(ALL_LDFLAGS)

clean:
	rm -r -f $(OBJS) nndep proj nndep_eigen *.o

.cpp.o: $(HEADERS) $(HEADERS_MAT) $(HEADERS_EIGEN)
	$(CC) -c $(ALL_CFLAGS) $<


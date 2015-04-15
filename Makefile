CC = g++

CFLAGS = -O3 -std=c++11 -I utils/ -Wall -funroll-loops
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

OBJS_FSC = $(OBJS) FSC_Classifier.o FSC_DependencyParser.o

HEADERS = Config.h Dataset.h DependencySent.h DependencyTree.h \
		  Configuration.h ParsingSystem.h ArcStandard.h Util.h \

HEADERS_MAT = Classifier.h DependencyParser.h
HEADERS_EIGEN = ClassifierEigen.h DependencyParserEigen.h

HEADERS_FSC = FSC_Classifier.h FSC_DependencyParser.h

all: nndep proj nndep_eigen nndep_fsc eval

eval : eval.o $(OBJS) $(HEADERS)
	$(CC) -o eval eval.o $(OBJS) $(CFLAGS)

proj : proj.o $(OBJS) $(HEADERS)
	$(CC) -o proj proj.o $(OBJS) $(CFLAGS)

nndep : nndep.o $(OBJS_MAT) $(HEADERS) $(HEADERS_MAT)
	$(CC) -o nndep nndep.o $(OBJS_MAT) $(CFLAGS) $(OMP_CFLAGS) $(LDFLAGS)

nndep_eigen : nndep_eigen.o $(OBJS_EIGEN) $(HEADERS) $(HEADERS_EIGEN)
	$(CC) -o nndep_eigen nndep_eigen.o $(OBJS_EIGEN) $(ALL_CFLAGS) $(ALL_LDFLAGS)

nndep_fsc : nndep_fsc.o $(OBJS_FSC) $(HEADERS) $(HEADERS_FSC)
	$(CC) -o nndep_fsc nndep_fsc.o $(OBJS_FSC) $(ALL_CFLAGS) $(ALL_LDFLAGS)

clean:
	rm -r -f $(OBJS) nndep proj nndep_eigen nndep_fsc *.o

.cpp.o: $(HEADERS) $(HEADERS_MAT) $(HEADERS_EIGEN) $(HEADERS_FSC)
	$(CC) -c $(ALL_CFLAGS) $<


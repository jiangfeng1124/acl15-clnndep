CC = g++ -std=c++11 -O3 -g

CFLAGS = -pthread -Wall -lm -fopenmp
INCLUDE = -I utils/ -I utils/eigen

OBJS = Config.o Dataset.o DependencySent.o DependencyTree.o \
	   Configuration.o ParsingSystem.o ArcStandard.o \
	   

OBJS_MAT = $(OBJS) Classifier.o DependencyParser.o
OBJS_EIGEN = $(OBJS) ClassifierEigen.o DependencyParserEigen.o

HEADERS = Config.h Dataset.h DependencySent.h DependencyTree.h \
		  Configuration.h ParsingSystem.h ArcStandard.h Util.h \

HEADERS_MAT = Classifier.h DependencyParser.h
HEADERS_EIGEN = ClassifierEigen.h DependencyParserEigen.h

all: nndep proj nndep_eigen

proj : proj.o $(OBJS) $(HEADERS)
	$(CC) -o proj proj.o $(OBJS) $(CFLAGS) $(INCLUDE)

nndep : nndep.o $(OBJS_MAT) $(HEADERS) $(HEADERS_MAT)
	$(CC) -o nndep nndep.o $(OBJS_MAT) $(CFLAGS) $(INCLUDE)

nndep_eigen : nndep_eigen.o $(OBJS_EIGEN) $(HEADERS) $(HEADERS_EIGEN)
	$(CC) -msse3 -o nndep_eigen nndep_eigen.o $(OBJS_EIGEN) $(CFLAGS) $(INCLUDE)

clean:
	rm -r -f $(OBJS) nndep proj nndep_eigen *.o

.cpp.o: $(HEADERS) $(HEADERS_MAT) $(HEADERS_EIGEN)
	$(CC) -c $(CFLAGS) $(INCLUDE) $<


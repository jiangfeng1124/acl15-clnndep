CC = g++ -std=c++11 -O3

CFLAGS = -pthread -Wall -lm
INCLUDE = -I utils/

OBJS = Config.o Dataset.o DependencySent.o DependencyTree.o \
	   Configuration.o ParsingSystem.o ArcStandard.o \
	   Classifier.o DependencyParser.o
HEADERS = Config.h Dataset.h DependencySent.h DependencyTree.h \
		  Configuration.h ParsingSystem.h ArcStandard.h \
		  Classifier.h DependencyParser.h Util.h

all: nndep proj

proj : proj.o $(OBJS) $(HEADERS)
	$(CC) -o proj proj.o $(OBJS) $(CFLAGS) $(INCLUDE)

nndep : nndep.o $(OBJS) $(HEADERs)
	$(CC) -o nndep nndep.o $(OBJS) $(CFLAGS) $(INCLUDE)

clean:
	rm -r -f $(OBJS) nndep proj *.o

.cpp.o: $(HEADERS)
	$(CC) -c $(CFLAGS) $(INCLUDE) $<


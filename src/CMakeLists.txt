include_directories (./utils)

set (clnndep_SRC
     ArcStandard.cpp
     ArcStandard.h
     Classifier.cpp
     Classifier.h
     Config.cpp
     Config.h
     Configuration.cpp
     Configuration.h
     Dataset.cpp
     Dataset.h
     DependencyParser.cpp
     DependencyParser.h
     DependencySent.cpp
     DependencySent.h
     DependencyTree.cpp
     DependencyTree.h
     fastexp.h
     nndep.cpp
     ParsingSystem.cpp
     ParsingSystem.h
     ThreadPool.h
     time.h
     Util.h)

add_executable (clnndep ${clnndep_SRC})


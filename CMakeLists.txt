cmake_minimum_required (VERSION 2.8.0)
project ("CLNNDEP - Cross-lingual Dependency Parsing")

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set (OMP true)
set (DEBUG true) # set false to speed up the program

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wno-narrowing -fpermissive -pthread -lm")

if (OMP)
    message (STATUS "OpenMP is on")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif (OMP)

if (NOT DEBUG)
    message (STATUS "Debug-mode is off")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
endif (NOT DEBUG)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set (SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set (CONFIGURE_DIR ${PROJECT_SOURCE_DIR}/conf)
set (RESOURCE_DIR ${PROJECT_SOURCE_DIR}/resources)
set (DATA_DIR ${PROJECT_SOURCE_DIR}/udt)

add_subdirectory (src)


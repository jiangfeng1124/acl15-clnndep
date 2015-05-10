cmake_minimum_required (VERSION 2.8.0)
project ("CLNNDEP - Cross-lingual Dependency Parsing")

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set (OMP true)

if (OMP)
    message (STATUS "OpenMP is on.")
    set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wno-narrowing -fpermissive -pthread -lm -fopenmp")
else (OMP)
    message (STATUS "OpenMP is off.")
    set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wno-narrowing -fpermissive -pthread -lm")
endif (OMP)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set (SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set (CONFIGURE_DIR ${PROJECT_SOURCE_DIR}/conf)
set (RESOURCE_DIR ${PROJECT_SOURCE_DIR}/resources)
set (DATA_DIR ${PROJECT_SOURCE_DIR}/udt)

add_subdirectory (src)


cmake_minimum_required(VERSION 3.1)
project(MNNTest)

find_package(OpenCV)
message("Opencv Version: ${OPENCV_VERSION}" )

file(GLOB LIBS lib/*.so)

set(INC_DIRS
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
)

set(SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/classifier.cpp
)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -O3 -std=c++11 -fopenmp")
include_directories(${INC_DIRS})
add_executable(classifier src/main.cpp ${SRC_FILES} )
target_link_libraries(classifier ${OpenCV_LIBS} ${LIBS})

cmake_minimum_required(VERSION 3.1)
project(MNNTest)

find_package(OpenCV)
message("Opencv Version: ${OPENCV_VERSION}" )

file(GLOB LIBS ${CMAKE_CURRENT_SOURCE_DIR}/../lib/*.so)

set(SRC_FILES
    ${PROJECT_SOURCE_DIR}/common/common.cpp
    ${PROJECT_SOURCE_DIR}/classifier/classifier.cpp
    ${PROJECT_SOURCE_DIR}/object/mobilenetssd.cpp
    ${PROJECT_SOURCE_DIR}/vision_engine.cpp
)
add_library(VisionEngine STATIC ${SRC_FILES})

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -O3 -std=c++11 -fopenmp")
target_link_libraries(VisionEngine PUBLIC ${LIBS} ${OpenCV_LIBS})

target_include_directories(VisionEngine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/classifier>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object>
)


if(MIRROR_BUILD_CLASSIFIER)
    add_executable(classifier ${CMAKE_CURRENT_SOURCE_DIR}/classifier/test_classifier.cpp)
    target_link_libraries(classifier PUBLIC VisionEngine)
endif()

if(MIRROR_BUILD_OBJECT)
    add_executable(object ${CMAKE_CURRENT_SOURCE_DIR}/object/test_object.cpp)
    target_link_libraries(object PUBLIC VisionEngine)
endif()

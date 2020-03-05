cmake_minimum_required(VERSION 3.1)
project(MNNTest)

find_package(OpenCV)
message("Opencv Version: ${OPENCV_VERSION}" )

file(GLOB LIBS ${CMAKE_CURRENT_SOURCE_DIR}/../lib/*.so)

set(SRC_FILES
    ${PROJECT_SOURCE_DIR}/common/common.cpp
    ${PROJECT_SOURCE_DIR}/classifier/classifier.cpp

    ${PROJECT_SOURCE_DIR}/object/mobilenetssd/mobilenetssd.cpp
    ${PROJECT_SOURCE_DIR}/object/object_engine.cpp

    ${PROJECT_SOURCE_DIR}/face/detecter/detecter.cpp
    ${PROJECT_SOURCE_DIR}/face/detecter/centerface/centerface.cpp
    ${PROJECT_SOURCE_DIR}/face/detecter/ultraface/ultraface.cpp
    ${PROJECT_SOURCE_DIR}/face/landmarker/landmarker.cpp
    ${PROJECT_SOURCE_DIR}/face/landmarker/pfldlandmarker/pfldlandmarker.cpp
    ${PROJECT_SOURCE_DIR}/face/landmarker/zqlandmarker/zqlandmarker.cpp
    ${PROJECT_SOURCE_DIR}/face/aligner/aligner.cpp
    ${PROJECT_SOURCE_DIR}/face/recognizer/recognizer.cpp
    ${PROJECT_SOURCE_DIR}/face/recognizer/mobilefacenet/mobilefacenet.cpp
    ${PROJECT_SOURCE_DIR}/face/database/face_database.cpp
    ${PROJECT_SOURCE_DIR}/face/database/stream/file_stream.cpp
    ${PROJECT_SOURCE_DIR}/face/face_engine.cpp

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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object/mobilenetssd>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detecter>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/centerface>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/ultraface>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/pfldlandmarker>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/zqlandmarker>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/aligner>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer/mobilefacenet>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/database>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/database/stream>
)

if(MIRROR_BUILD_CLASSIFIER)
    add_executable(classifier ${CMAKE_CURRENT_SOURCE_DIR}/classifier/test_classifier.cpp)
    target_link_libraries(classifier PUBLIC VisionEngine)
endif()

if(MIRROR_BUILD_OBJECT)
    add_executable(object ${CMAKE_CURRENT_SOURCE_DIR}/object/test_object.cpp)
    target_link_libraries(object PUBLIC VisionEngine)
endif()

if(MIRROR_BUILD_FACE)
    add_executable(face ${CMAKE_CURRENT_SOURCE_DIR}/face/test_face.cpp)
    target_link_libraries(face PUBLIC VisionEngine)
endif()


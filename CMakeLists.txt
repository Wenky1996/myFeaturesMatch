cmake_minimum_required(VERSION 3.15)
project(FeatureMatchEvaluation)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV 3.4 REQUIRED)

include_directories(${OpenCV_INCLUDE_LIBS})


add_executable(FeatureMatchEvaluation main.cpp)
target_link_libraries(FeatureMatchEvaluation ${OpenCV_LIBS})
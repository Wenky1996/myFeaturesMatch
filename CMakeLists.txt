cmake_minimum_required(VERSION 3.15)
project(FeatureMatchEvaluation)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV 3.4 REQUIRED)

include_directories(${OpenCV_INCLUDE_LIBS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(AgainstNuclearCorner
        src/AgainstNuclearCorner.cpp)

add_executable(MatchEvaluation  test/main.cpp)
add_executable(TEST test/test.cpp include/AgainstNuclearCorner.h src/AgainstNuclearCorner.cpp)

target_link_libraries(MatchEvaluation ${OpenCV_LIBS} AgainstNuclearCorner)
target_link_libraries(TEST ${OpenCV_LIBS})



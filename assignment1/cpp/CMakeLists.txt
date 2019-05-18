cmake_minimum_required(VERSION 3.5)

PROJECT(IncameraTest)

include_directories(${CMAKE_SOURCE_DIR}/incamera/include)
ADD_SUBDIRECTORY(incamera)

ADD_EXECUTABLE(IncameraTest main.cpp)
TARGET_LINK_LIBRARIES(IncameraTest
    IncameraProcessor)

SET_TARGET_PROPERTIES(IncameraTest PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")






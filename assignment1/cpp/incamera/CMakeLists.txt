include(ExternalProject)

find_package( OpenCV REQUIRED  )

include_directories(${CMAKE_SOURCE_DIR}/incamera/include/)

message('${OpenCV_INCLUDE_DIRS}')

include_directories( ${OpenCV_INCLUDE_DIRS}  )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


add_library(IncameraProcessor
    SHARED    
    incamera_processor.cpp)

set_target_properties(IncameraProcessor PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/incamera/bin/libincamera.so")
target_link_libraries(IncameraProcessor ${OpenCV_LIBS})



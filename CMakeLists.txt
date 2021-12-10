cmake_minimum_required(VERSION 3.20)
project(Coursework_materials)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR /usr/local/Cellar/opencv@2/2.4.13.7_12/include/opencv)

include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( OpenCV REQUIRED )

add_executable(noEntryDetector noEntryDetector.cpp)

target_link_libraries(noEntryDetector ${OpenCV_LIBS} )

add_executable(Coursework_materials
        noEntryDetector.cpp)

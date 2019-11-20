cmake_minimum_required(VERSION 2.8)
project(MyOpenCVProject)

# Create Main project
set(SOURCES
    annotation.cpp
)

set(HEADERS
    # main.h
)

add_executable(annotation
    ${SOURCES}
    ${HEADERS}
)


# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    target_include_directories(annotation PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(annotation ${OpenCV_LIBS})
endif()
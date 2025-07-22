cmake_minimum_required(VERSION 3.14)
project(zed_foxglove LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find ZED SDK
find_package(ZED 5 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)

# Fetch Foxglove SDK
include(FetchContent)
FetchContent_Declare(
    foxglove
    # See available releases and builds here: https://github.com/foxglove/foxglove-sdk/releases
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2Fv0.9.0/foxglove-v0.9.0-cpp-x86_64-unknown-linux-gnu.zip
)
FetchContent_MakeAvailable(foxglove)

# Add executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Add include directory for Foxglove SDK
target_include_directories(${PROJECT_NAME} PRIVATE ${foxglove_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${foxglove_SOURCE_DIR}/include/foxglove)

# Include OpenCV directories
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# Include ZED SDK directories
target_include_directories(${PROJECT_NAME} PRIVATE ${ZED_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

# Find all Foxglove SDK source files
file(GLOB FOXGLOVE_SOURCES CONFIGURE_DEPENDS
    "${foxglove_SOURCE_DIR}/src/*.cpp"
    "${foxglove_SOURCE_DIR}/src/server/*.cpp"
)

# Add Foxglove SDK source files
target_sources(${PROJECT_NAME} PRIVATE ${FOXGLOVE_SOURCES})

# Link against libfoxglove.a
target_link_libraries(${PROJECT_NAME} PRIVATE ${foxglove_SOURCE_DIR}/lib/libfoxglove.a)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Link ZED SDK libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${ZED_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})

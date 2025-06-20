cmake_minimum_required(VERSION 3.14)
project(quickstart LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use LLVM's libc++ instead of GCC's libstdc++
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++)
endif()

# Fetch Foxglove SDK
include(FetchContent)
FetchContent_Declare(
    foxglove
    # See available releases and builds here: https://github.com/foxglove/foxglove-sdk/releases
    URL https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2Fv0.9.0/foxglove-v0.9.0-cpp-x86_64-unknown-linux-gnu.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(foxglove)

# Add executable
add_executable(quickstart src/main.cpp)

# Add include directory for Foxglove SDK
target_include_directories(quickstart PRIVATE ${foxglove_SOURCE_DIR}/include)
target_include_directories(quickstart PRIVATE ${foxglove_SOURCE_DIR}/include/foxglove)

# Find all Foxglove SDK source files
file(GLOB FOXGLOVE_SOURCES CONFIGURE_DEPENDS
    "${foxglove_SOURCE_DIR}/src/*.cpp"
    "${foxglove_SOURCE_DIR}/src/server/*.cpp"
)

# Add Foxglove SDK source files
target_sources(quickstart PRIVATE ${FOXGLOVE_SOURCES})

# Link against libfoxglove.a
target_link_libraries(quickstart PRIVATE ${foxglove_SOURCE_DIR}/lib/libfoxglove.a)

cmake_minimum_required(VERSION 3.10)
project(LampTensor VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- NEW: Set output directory for executables ---
# ${CMAKE_BINARY_DIR} refers to your 'build' folder.
# This puts executables in build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(include)

# 1. Define the variable with a default value
set(TEST_TARGET "test_linear" CACHE STRING "The test file to compile (test_tensor or test_linear)")

# 2. Always build the core library
add_library(lamp 
    src/tensor.cpp 
    src/linear.cpp
    src/module.cpp
    src/sequential.cpp
    src/optimizer.cpp
)

# 3. Check the variable and choose the source file
if(TEST_TARGET STREQUAL "test_tensor")
    message(STATUS "Compiling Target: test_tensor.cpp")
    add_executable(run_test examples/test_tensor.cpp)

elseif(TEST_TARGET STREQUAL "test_linear")
    message(STATUS "Compiling Target: test_linear.cpp")
    add_executable(run_test examples/test_linear.cpp)

elseif(TEST_TARGET STREQUAL "test_autograd")
    message(STATUS "Compiling Target: test_autograd.cpp")
    add_executable(run_test examples/test_autograd.cpp)

else()
    message(FATAL_ERROR "Unknown TEST_TARGET: ${TEST_TARGET}. Valid options are: test_tensor, test_linear")
endif()

# 4. Link the chosen executable to the library
target_link_libraries(run_test PRIVATE lamp)
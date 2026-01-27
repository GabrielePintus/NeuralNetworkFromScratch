cmake_minimum_required(VERSION 3.10)
project(LampTensor VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Include the header directory globally for this project
include_directories(include)

# 2. Create the library from source in src/
add_library(lamp src/tensor.cpp)

# 3. Create the test executable
add_executable(test_tensor examples/test_tensor.cpp)

# 4. Link the library to the executable
target_link_libraries(test_tensor PRIVATE lamp)
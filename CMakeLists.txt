cmake_minimum_required(VERSION 3.10)
project(Lamp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- NEW: Set output directory for executables ---
# ${CMAKE_BINARY_DIR} refers to your 'build' folder.
# This puts executables in build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(include)

# 2. Always build the core library
add_library(lamp
    src/core/tensor.cpp
    src/nn/linear.cpp
    src/nn/module.cpp
    src/nn/sequential.cpp
    src/optim/optimizer.cpp
    src/nn/losses.cpp
    src/nn/activations.cpp
    src/data/data_utils.cpp
    src/data/dataloader.cpp
)

add_executable(test_tensor examples/test_tensor.cpp)
add_executable(test_linear examples/test_linear.cpp)
add_executable(test_autograd examples/test_autograd.cpp)

# 4. Link the chosen executable to the library
target_link_libraries(test_tensor PRIVATE lamp)
target_link_libraries(test_linear PRIVATE lamp)
target_link_libraries(test_autograd PRIVATE lamp)

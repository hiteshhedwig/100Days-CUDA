cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

project(100dayscuda LANGUAGES CXX CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/include)

# Add common library
add_subdirectory(common)

# Add days subdirectories
add_subdirectory(days/day001)
add_subdirectory(days/day002)
add_subdirectory(days/day003)
add_subdirectory(days/day004)
add_subdirectory(days/day005)
add_subdirectory(days/day006)
add_subdirectory(days/day007)
add_subdirectory(days/day008)


# Add projects subdirectory
# Uncomment when you start working on projects:
# add_subdirectory(projects)

# Add tests subdirectory
# Uncomment when you start writing tests:
# add_subdirectory(tests)

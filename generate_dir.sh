#!/bin/bash

# Create main README.md
cat > README.md << 'EOF'
# 100 Days of CUDA Learning

This repository contains my journey of learning CUDA over 100 days.

## Project Structure

- `common/`: Common utility code used across multiple days
- `days/`: Individual daily exercises and experiments
- `projects/`: Larger mini-projects spanning multiple days
- `data/`: Sample data for projects
- `tests/`: Unit tests
- `docs/`: Documentation and learning notes

## Progress

| Day | Topic | Status |
|-----|-------|--------|
| 001 | Vector Addition | Completed |
| 002 | Matrix Multiplication | Planned |
| ... | ... | ... |

EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Build directories
build/
debug/
release/
*.o
*.obj
*.exe
*.out
*.app
*.so
*.a
*.dll

# CUDA specific
*.i
*.ii
*.gpu
*.ptx
*.cubin
*.fatbin

# IDE files
.vscode/
.idea/
*.suo
*.user
*.sln.docstates
*.sdf
*.opensdf
*.VC.db
*.VC.opendb

# CMake
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
cmake-build-*/

# Temporary files
*~
*.swp
*.bak
EOF

# Create main CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
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
# Uncomment as you progress:
# add_subdirectory(days/day002)
# ...

# Add projects subdirectory
# Uncomment when you start working on projects:
# add_subdirectory(projects)

# Add tests subdirectory
# Uncomment when you start writing tests:
# add_subdirectory(tests)
EOF

# Create common directory structure
mkdir -p common/include common/src

# Create cuda_utils.cuh
cat > common/include/cuda_utils.cuh << 'EOF'
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Device query function
void printDeviceInfo();

#endif // CUDA_UTILS_CUH
EOF

# Create timer.cuh
cat > common/include/timer.cuh << 'EOF'
#ifndef TIMER_CUH
#define TIMER_CUH

#include <cuda_runtime.h>

// Start CUDA timer
void startTimer(cudaEvent_t *start, cudaEvent_t *stop);

// Stop CUDA timer and return elapsed time in milliseconds
float stopTimer(cudaEvent_t start, cudaEvent_t stop);

#endif // TIMER_CUH
EOF

# Create data_utils.cuh
cat > common/include/data_utils.cuh << 'EOF'
#ifndef DATA_UTILS_CUH
#define DATA_UTILS_CUH

// Initialize array with random values
void initializeArray(float *arr, int size);

// Verify results of vector addition
bool verifyVectorAdd(const float *A, const float *B, const float *C, int size);

#endif // DATA_UTILS_CUH
EOF

# Create cuda_utils.cu
cat > common/src/cuda_utils.cu << 'EOF'
#include "cuda_utils.cuh"

void printDeviceInfo() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("No CUDA devices found\n");
        return;
    }
    
    printf("Found %d CUDA device(s):\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, i));
        
        printf("\nDevice %d: \"%s\"\n", i, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %.2f GB\n", 
               (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        printf("  Max dimensions of a block: (%d, %d, %d)\n", 
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max dimensions of a grid: (%d, %d, %d)\n", 
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }
    
    printf("\n");
}
EOF

# Create timer.cu
cat > common/src/timer.cu << 'EOF'
#include "timer.cuh"
#include "cuda_utils.cuh"

void startTimer(cudaEvent_t *start, cudaEvent_t *stop) {
    CUDA_CHECK(cudaEventCreate(start));
    CUDA_CHECK(cudaEventCreate(stop));
    CUDA_CHECK(cudaEventRecord(*start));
}

float stopTimer(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return milliseconds;
}
EOF

# Create data_utils.cu
cat > common/src/data_utils.cu << 'EOF'
#include "data_utils.cuh"
#include <stdlib.h>
#include <math.h>

void initializeArray(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() / (float)RAND_MAX;
    }
}

bool verifyVectorAdd(const float *A, const float *B, const float *C, int size) {
    const float epsilon = 1e-5;
    for (int i = 0; i < size; i++) {
        if (fabs((A[i] + B[i]) - C[i]) > epsilon) {
            return false;
        }
    }
    return true;
}
EOF

# Create common CMakeLists.txt
cat > common/CMakeLists.txt << 'EOF'
add_library(cuda_utils
    src/cuda_utils.cu
    src/timer.cu
    src/data_utils.cu
)

target_include_directories(cuda_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(cuda_utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
EOF

# Create days directories
mkdir -p days

# Create directories for all 100 days
for i in $(seq -f "%03g" 1 100); do
    mkdir -p "days/day$i"
done

# Create day001 vector_add.cu
cat > days/day001/vector_add.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "timer.cuh"
#include "data_utils.cuh"

// Simple CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    // Print device information
    printDeviceInfo();
    
    // Problem size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("Vector addition of %d elements\n", numElements);
    
    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize host arrays
    initializeArray(h_A, numElements);
    initializeArray(h_B, numElements);
    
    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));
    
    // Transfer data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Set up execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    // Create timing events
    cudaEvent_t start, stop;
    startTimer(&start, &stop);
    
    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Record execution time
    float milliseconds = stopTimer(start, stop);
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Verify results
    bool correctResult = verifyVectorAdd(h_A, h_B, h_C, numElements);
    printf("Test %s\n", correctResult ? "PASSED" : "FAILED");
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Done\n");
    return 0;
}
EOF

# Create day001 CMakeLists.txt
cat > days/day001/CMakeLists.txt << 'EOF'
add_executable(day001_vector_add vector_add.cu)
target_link_libraries(day001_vector_add cuda_utils)
set_target_properties(day001_vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
EOF

# Create day001 README.md
cat > days/day001/README.md << 'EOF'
# Day 1: Vector Addition

## Today's Goals
- Set up CUDA development environment
- Understand basic CUDA kernel launch syntax
- Implement and test vector addition kernel
- Learn about error handling in CUDA
- Measure kernel execution time

## Notes
- The `blockDim.x * blockIdx.x + threadIdx.x` formula gives the global thread ID
- Error checking is important for CUDA operations
- Memory transfers between host and device can be a performance bottleneck

## Performance Results
- Execution time: [Add your measurements here]
- Throughput: [Add your calculations here]

## Questions to Explore
- How does changing the block size affect performance?
- What happens if we use too many threads per block?
- How does the performance compare with CPU implementation?
EOF

# Create projects directories
mkdir -p projects/image_processing/include projects/image_processing/src
mkdir -p projects/neural_network/include projects/neural_network/src

# Create data directories
mkdir -p data/images data/datasets

# Create tests directory
mkdir -p tests

# Create tests CMakeLists.txt
cat > tests/CMakeLists.txt << 'EOF'
add_executable(test_utils test_utils.cu)
target_link_libraries(test_utils cuda_utils)
set_target_properties(test_utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
EOF

# Create docs directory
mkdir -p docs/images

# Create learning_path.md
cat > docs/learning_path.md << 'EOF'
# CUDA Learning Path

## Weeks 1-2: CUDA Basics
- Day 001-005: CUDA programming model, memory hierarchy, simple vector operations
- Day 006-010: Matrix operations, thread synchronization, error handling

## Weeks 3-4: Memory Management
- Day 011-015: Global, shared, constant memory
- Day 016-020: Texture memory, pinned memory
- Day 021-025: Unified memory, memory coalescing

## Weeks 5-7: Performance Optimization
- Day 026-030: Coalesced memory access
- Day 031-035: Bank conflicts
- Day 036-040: Occupancy optimization
- Day 041-045: Warp divergence

## Weeks 8-10: Advanced Topics
- Day 046-050: Streams and asynchronous operations
- Day 051-055: Dynamic parallelism
- Day 056-060: Multi-GPU programming
- Day 061-065: Thrust library
- Day 066-070: cuBLAS, cuDNN

## Weeks 11-14: Projects
- Day 071-080: Image processing application
- Day 081-090: Neural network implementation
- Day 091-100: Final project and optimization
EOF

# Create cuda_notes.md
cat > docs/cuda_notes.md << 'EOF'
# CUDA Notes

## CUDA Programming Model
- CUDA is an extension to C/C++ for parallel programming on NVIDIA GPUs
- **Thread Hierarchy**:
  - Thread: Single execution unit
  - Block: Group of threads that can cooperate
  - Grid: Group of blocks
- **Memory Hierarchy**:
  - Register: Per-thread
  - Shared Memory: Per-block
  - Global Memory: Accessible by all threads
  - Constant Memory: Read-only for all threads
  - Texture Memory: Specialized for spatial locality

## CUDA Syntax
- `__global__`: Function called from CPU, runs on GPU
- `__device__`: Function called from GPU, runs on GPU
- `__host__`: Function called from CPU, runs on CPU
- `<<<gridDim, blockDim>>>`: Kernel launch configuration

## Best Practices
- Use shared memory for frequently accessed data
- Minimize data transfer between CPU and GPU
- Ensure coalesced memory access
- Avoid warp divergence
- Balance occupancy and resource usage
EOF

# Create resources.md
cat > docs/resources.md << 'EOF'
# CUDA Learning Resources

## Official Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

## Books
- "CUDA by Example" by Jason Sanders and Edward Kandrot
- "Programming Massively Parallel Processors" by David B. Kirk and Wen-mei W. Hwu

## Online Courses
- [Udacity: Intro to Parallel Programming](https://www.udacity.com/course/intro-to-parallel-programming--cs344)
- [Coursera: Heterogeneous Parallel Programming](https://www.coursera.org/learn/heterogeneous-parallel-programming)

## Tutorials
- [NVIDIA CUDA Tutorial](https://developer.nvidia.com/cuda-education)
- [CUDA Crash Course](https://github.com/CoffeeBeforeArch/cuda_programming)

## Tools
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)

## Communities
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/159)
- [Stack Overflow CUDA tag](https://stackoverflow.com/questions/tagged/cuda)
EOF

echo "CUDA 100-days project structure has been created!"
echo "Navigate to the cuda-100days directory to start your learning journey."

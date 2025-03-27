#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>  // For CPU timing
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

// CPU implementation of vector addition
void vectorAddCPU(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; i++) {
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
    float *h_C_CPU = (float *)malloc(size);  // For CPU results
    
    if (h_A == NULL || h_B == NULL || h_C == NULL || h_C_CPU == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize host arrays
    initializeArray(h_A, numElements);
    initializeArray(h_B, numElements);
    
    //-------------------------------------------------------------------------
    // CPU Implementation
    //-------------------------------------------------------------------------
    printf("\n--- CPU Implementation ---\n");
    
    // Start CPU timer
    clock_t cpu_start = clock();
    
    // Run CPU implementation
    vectorAddCPU(h_A, h_B, h_C_CPU, numElements);
    
    // Stop CPU timer and calculate time
    clock_t cpu_end = clock();
    double cpu_time_ms = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU execution time: %.3f ms\n", cpu_time_ms);
    
    //-------------------------------------------------------------------------
    // GPU Implementation
    //-------------------------------------------------------------------------
    printf("\n--- GPU Implementation ---\n");
    
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
    int threadsPerBlock = 512;
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
    float gpu_time_ms = stopTimer(start, stop);
    printf("GPU kernel execution time: %.3f ms\n", gpu_time_ms);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    //-------------------------------------------------------------------------
    // Verification and Performance Comparison
    //-------------------------------------------------------------------------
    // Verify GPU results
    bool gpuCorrect = verifyVectorAdd(h_A, h_B, h_C, numElements);
    printf("GPU Test %s\n", gpuCorrect ? "PASSED" : "FAILED");
    
    // Verify CPU results
    bool cpuCorrect = verifyVectorAdd(h_A, h_B, h_C_CPU, numElements);
    printf("CPU Test %s\n", cpuCorrect ? "PASSED" : "FAILED");
    
    // Compare CPU and GPU results
    bool resultsMatch = true;
    for (int i = 0; i < numElements; i++) {
        if (h_C[i] != h_C_CPU[i]) {
            resultsMatch = false;
            printf("Results mismatch at element %d: GPU=%.1f, CPU=%.1f\n", 
                  i, h_C[i], h_C_CPU[i]);
            break;
        }
    }
    printf("CPU and GPU results %s\n", resultsMatch ? "MATCH" : "DIFFER");
    
    // Calculate and display speedup
    if (cpu_time_ms > 0) {
        printf("\n--- Performance Comparison ---\n");
        printf("Speedup (CPU/GPU): %.2fx\n", cpu_time_ms / gpu_time_ms);
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);
    
    printf("Done\n");
    return 0;
}
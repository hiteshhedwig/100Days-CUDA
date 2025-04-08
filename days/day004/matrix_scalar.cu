#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <string>
#include <fstream>
#include "data_utils.cuh"

// CUDA kernel 
__global__ void matrixScalarMul(const float *A, float *C, 
                                int numElements, float factor) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] * factor;
        // printf("Multiplying factor %0.3f with %0.3f = %0.3f \n", factor, A[i], C[i]);
    }
}

// CPU implementation 
void matrixScalarMul_CPU( const float *A, float *C, 
                          int numElements, float factor) {
    for (int i = 0; i < numElements; i++) {
        C[i] = factor*A[i];
        // printf("Multiplying factor %0.3f with %0.3f = %0.3f \n", factor, A[i], C[i]);
    }
}

// Measure CPU performance
double benchmarkCPU(const float *A, float *C, 
                    int numElements, int iterations, 
                    float factor) {

    clock_t start = clock();
    
    for (int iter = 0; iter < iterations; iter++) {
        matrixScalarMul_CPU(A, C, numElements, factor);
    }
    
    clock_t end = clock();
    return 1000.0 * (end - start) / CLOCKS_PER_SEC / iterations;
}

// Measure GPU performance
double benchmarkGPU(const float *h_A, float *h_C, int numElements, int iterations, float factor) {
    size_t size = numElements * sizeof(float);
    
    // Allocate device memory
    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    
    // Set kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    // printf("Warmup! \n");
    matrixScalarMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements, factor);
    cudaDeviceSynchronize();
    
    float totalTime = 0.0f;
    
    // Benchmark iterations
    // printf("Real work! \n");
    cudaEventRecord(start);
    for (int iter = 0; iter < iterations; iter++) {
        matrixScalarMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements, factor);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&totalTime, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return totalTime / iterations;
}

int main() {
    // Output file for results
    std::ofstream resultsFile("benchmark_results_scalar_mul.csv");
    resultsFile << "MatrixSize,TotalElements,CPU_Time_ms,GPU_Time_ms,Speedup\n";
    
    // Test different matrix sizes
    std::vector<int> matrixSizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int iterations = 10; // Number of iterations for each test
    float factor = 3.141592;
    
    for (int size : matrixSizes) {
        int totalElements = size * size;
        size_t memSize = totalElements * sizeof(float);
        
        printf("Benchmarking matrix size: %d x %d (%d elements)\n", size, size, totalElements);
        
        // Allocate and initialize host memory
        float *h_A = (float *)malloc(memSize);
        float *h_C_cpu = (float *)malloc(memSize);
        float *h_C_gpu = (float *)malloc(memSize);
        
        // Initialize data
        for (int i = 0; i < totalElements; i++) {
            h_A[i] = rand() / (float)RAND_MAX;
        }
        
        // printLinearMatrix(h_A, size);

        // Run CPU benchmark
        double cpuTime = benchmarkCPU(h_A, h_C_cpu, totalElements, 
                                      iterations, factor);
        printf("  CPU time: %.3f ms\n", cpuTime);
        
        // Run GPU benchmark
        double gpuTime = benchmarkGPU(h_A, h_C_gpu, totalElements, 
                                      iterations, factor);
        printf("  GPU time: %.3f ms\n", gpuTime);
        
        // Calculate speedup
        double speedup = cpuTime / gpuTime;
        printf("  Speedup: %.2fx\n\n", speedup);
        
        // Write results to file
        resultsFile << size << "," << totalElements << "," 
                   << cpuTime << "," << gpuTime << "," << speedup << "\n";
        
        // Verify results
        bool correct = true;
        for (int i = 0; i < totalElements; i++) {
            if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5) {
                printf("Results do not match at element %d! CPU: %f, GPU: %f\n", 
                       i, h_C_cpu[i], h_C_gpu[i]);
                correct = false;
                break;
            }
        }
        if (correct) {
            printf("  Results verified: GPU and CPU outputs match\n");
        }
        
        // Free memory
        free(h_A);
        free(h_C_cpu);
        free(h_C_gpu);
    }
    
    resultsFile.close();
    printf("Benchmark complete. Results saved to benchmark_results_add.csv\n");
    
    return 0;
}
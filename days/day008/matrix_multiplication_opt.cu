#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <string>
#include <fstream>
// #include "data_utils.cuh"
#include "cuda_utils.cuh"

#define TILE_SIZE 2

// Function to print a matrix stored in row-major format
void printMatrix(const float* matrix, int size) {
    printf("Matrix (%dx%d):\n", size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%8.4f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Function to print a vector
void printVector(const float* vector, int size) {
    printf("Vector (%d):\n", size);
    for (int i = 0; i < size; i++) {
        printf("%8.4f ", vector[i]);
        // Optional: Add a newline every 8 or 10 elements for readability
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("\n\n");
}

__global__ void matrixMul_opt(const float *A, const float *B, float *C, int numElements) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y; 
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by*TILE_SIZE + ty;
    int col = bx*TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Process the matrix in tiles
    for (int tile = 0; tile < (numElements) / TILE_SIZE; tile++) {
        // Load tile from A - standard access pattern
        sharedA[ty][tx] = A[row*numElements + (tile*TILE_SIZE + tx)];
        
        // Load tile from B - transpose access pattern to match CPU implementation
        sharedB[ty][tx] = B[(tile*TILE_SIZE + tx)*numElements + row];

        __syncthreads();

        // Compute partial dot product - modified to account for transposed B
        for (int k = 0; k < TILE_SIZE; k++)
            sum += sharedA[ty][k] * sharedB[tx][k]; // Note the swapped indices
            
        __syncthreads();
    }
    
    if (row < numElements && col < numElements) {
        C[row*numElements+col] = sum;
    }
}


// CPU implementation -
void matrixMul_CPU(const float *A, const float *B, float *C, 
                          int numElements) {
    
    for(int row_fixed=0; row_fixed<numElements; row_fixed++) {
        // iterate over row and cols of second matrix
        // printf("idx of row fixed [%d] \n", row_fixed);
        for (int row_idx=0; row_idx<numElements; row_idx++) {
            // now cols
            float sum_local=0;
            for (int col_idx=0; col_idx<numElements; col_idx++){
                // printf("idx of row[%d]  - col[%d]  -- [%d] \n", row_fixed, col_idx, row_idx);
                // printf("multiplication of row x col = [%.3f] x [%.3f] \n", 
                //                                       A[row_fixed*numElements + col_idx], 
                //                                       B[col_idx*numElements + row_idx]
                //                                       );
                sum_local += A[row_fixed*numElements + col_idx] * B[col_idx*numElements + row_idx];
                // printf("SUM_local  - [%.4f] \n", sum_local );
            }
            C[row_fixed*numElements + row_idx] = sum_local ;
            // printf("Indices - [row_fixed] - [[%d]] \n ", row_fixed);
            // printf("Total sum_local  - [%d] -- C [%0.3f] \n", row_fixed*numElements + row_idx , C[row_fixed*numElements + row_idx]);
        }
    }

}

// Measure CPU performance
double benchmarkCPU(const float *A, const float *B, float *C, 
                    int size, int iterations) {
    clock_t start = clock();
    
    for (int iter = 0; iter < iterations; iter++) {
        matrixMul_CPU(A, B, C, size);
    }
    
    clock_t end = clock();
    return 1000.0 * (end - start) / CLOCKS_PER_SEC / iterations;
}

// Measure GPU performance
double benchmarkGPU(const float *h_A, const float *h_B, float *h_C, 
                    int size, int iterations) {
    size_t matrixSize = size * size * sizeof(float);
    size_t matrixSize2 = size * size * sizeof(float);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize2);
    cudaMalloc((void **)&d_C, matrixSize2);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize2, cudaMemcpyHostToDevice);
    
    // Set kernel configuration
    dim3 threadsPerBlock (TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMul_opt<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    matrixMul_opt<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();
    
    float totalTime = 0.0f;
    
    // Benchmark iterations
    cudaEventRecord(start);
    for (int iter = 0; iter < iterations; iter++) {
        matrixMul_opt<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&totalTime, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, matrixSize2, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return totalTime / iterations;
}

int main() {
    // Output file for results
    std::ofstream resultsFile("benchmark_results_mat_mat_mul_opt.csv");
    resultsFile << "MatrixSize,TotalElements,CPU_Time_ms,GPU_Time_ms,Speedup\n";
    
    // Test different matrix sizes
    // std::vector<int> matrixSizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<int> matrixSizes = {4};  // Uncomment for quick testing

    int iterations = 1; // Number of iterations for each test
    
    for (int size : matrixSizes) {
        int totalElements = size * size;
        size_t matrixSize = totalElements * sizeof(float);
        size_t matrixSize2 = totalElements * sizeof(float);
        
        printf("Benchmarking matrix-matrix multiplication: %d x %d matrix (%d elements)\n", 
               size, size, totalElements);
        
        // Allocate and initialize host memory
        float *h_A = (float *)malloc(matrixSize);       // Matrix
        float *h_B = (float *)malloc(matrixSize2);       // Input vector
        float *h_C_cpu = (float *)malloc(matrixSize2);   // Result vector
        float *h_C_gpu = (float *)malloc(matrixSize2);   // Result vector
        
        // Initialize matrix A with random values
        for (int i = 0; i < totalElements; i++) {
            h_A[i] = rand() / (float)RAND_MAX;
        }
        
        // Initialize matrix B with random values
        for (int i = 0; i < totalElements; i++) {
            h_B[i] = rand() / (float)RAND_MAX;
        }

        printf("Sizes : %d \n ", size );
        if (size <= 16) {  // Only print for small matrices to avoid flooding the console
            printMatrix(h_A, size);
            printMatrix(h_B, size);
        }

        
        // Run CPU benchmark
        double cpuTime = benchmarkCPU(h_A, h_B, h_C_cpu, size, iterations);
        printf("  CPU time: %.3f ms\n", cpuTime);

        printMatrix(h_C_cpu, size);
        
        // Run GPU benchmark
        double gpuTime = benchmarkGPU(h_A, h_B, h_C_gpu, size, iterations);
        printf("  GPU time: %.3f ms\n", gpuTime);

        printMatrix(h_C_gpu, size);

        
        // Calculate speedup
        double speedup = cpuTime / gpuTime;
        printf("  Speedup: %.2fx\n\n", speedup);
        
        // Write results to file
        resultsFile << size << "," << totalElements << "," 
                   << cpuTime << "," << gpuTime << "," << speedup << "\n";
        
        // Verify results
        bool correct = true;
        for (int i = 0; i < size*size; i++) {
            if (fabs(h_C_cpu[i] - h_C_gpu[i]) / fabs(h_C_cpu[i]) > 1e-6) {   
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
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
    }
    
    resultsFile.close();
    printf("Benchmark complete. Results saved to benchmark_results_mat_vec_mul_opt.csv\n");
    
    return 0;
}
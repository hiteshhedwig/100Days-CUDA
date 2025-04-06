#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "timer.cuh"
#include "data_utils.cuh"

__global__ void matrixMulKernel(float *A, float *B, float *C, int numElements) {
    // Calculate which result element this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check boundaries
    if (row < numElements && col < numElements) {
        // Each thread computes one element of C
        float sum = 0.0f;
        
        // Dot product of row from A and column from B
        for (int k = 0; k < numElements; k++) {
            sum += A[row * numElements + k] * B[k * numElements + col];
        }
        
        // Store the result
        C[row * numElements + col] = sum;
    }
}

void vectorMulCPU(const float *A, const float *B, float *C, int numElements) {
    // basic cpu based multiplication
    // iterate over a fixed row
    for(int row_fixed=0; row_fixed<numElements; row_fixed++) {
        // iterate over row and cols of second matrix
        printf("idx of row fixed [%d] \n", row_fixed);
        for (int row_idx=0; row_idx<numElements; row_idx++) {
            // now cols
            float sum_local =0;
            for (int col_idx=0; col_idx<numElements; col_idx++){
                printf("idx of row[%d]  - col[%d]  -- [%d] \n", row_fixed, col_idx, row_idx);
                printf("multiplication of row x col = [%.3f] x [%.3f] \n", 
                                                      A[row_fixed*numElements + col_idx], 
                                                      B[col_idx*numElements + row_idx]
                                                      );

                sum_local  = sum_local  + A[row_fixed*numElements + col_idx] * B[col_idx*numElements + row_idx];
                printf("SUM_local  - [%.4f] \n", sum_local );
            }
            C[row_fixed*numElements + row_idx] = sum_local ;
            printf("Indices - [row_fixed*numElements + row_idx] - [[%d]*[%d] + [%d]] \n ", row_fixed, numElements, row_idx);
            printf("Total sum_local  - [%d] -- C [%0.3f] \n", sum_local , C[row_fixed*numElements + row_idx]);
        }
    }
}

void printMatrix(float **matrix, int size) {
    printf("Matrix:\n");
    for (int i = 0; i < size; i++) {
        printf("  Row %d:", i);
        for (int j = 0; j < size; j++) {
            printf("  %.3f", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printLinearMatrix(float *linearMatrix, int size) {
    printf("Matrix:\n");
    for (int i = 0; i < size; i++) {
        printf("  Row %d:", i);
        for (int j = 0; j < size; j++) {
            // Calculate the 1D index from 2D coordinates using the same formula
            int index = i * size + j;
            printf("  %.3f", linearMatrix[index]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    // Print device information
    printDeviceInfo();
    
    // Problem size
    int numElements = 3;
    printf("Vector multiplication of %d elements\n", numElements);

    size_t size = numElements * numElements * sizeof(float);
    
    // Allocate host memory
    float **h_A = (float **)malloc(numElements * sizeof(float *));
    float **h_B = (float **)malloc(numElements * sizeof(float *));
    float **h_C = (float **)malloc(numElements * sizeof(float *));
    
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; i++) {
        h_A[i] = (float *)malloc(numElements * sizeof(float));
        h_B[i] = (float *)malloc(numElements * sizeof(float));
        h_C[i] = (float *)malloc(numElements * sizeof(float));
    }

    
    // Initialize host arrays
    initializeMatrix(h_A, numElements);
    initializeMatrix(h_B, numElements);

    printMatrix(h_A, numElements);

    printMatrix(h_B, numElements);

    // Create linear arrays for CUDA
    float *h_A_linear = (float *)malloc(size);
    float *h_B_linear = (float *)malloc(size);
    float *h_C_linear = (float *)malloc(size);
    
    if (h_A_linear == NULL || h_B_linear == NULL || h_C_linear == NULL) {
        fprintf(stderr, "Failed to allocate host memory for linear arrays\n");
        exit(EXIT_FAILURE);
    }

    // Copy from 2D to linear arrays
    for (int i = 0; i < numElements; i++) {
        for (int j = 0; j < numElements; j++) {
            h_A_linear[i * numElements + j] = h_A[i][j];
            h_B_linear[i * numElements + j] = h_B[i][j];
            h_C_linear[i * numElements + j] = -1;
        }
    }

    //-------------------------------------------------------------------------
    // CPU Implementation
    //-------------------------------------------------------------------------
    printf("\n--- CPU Implementation ---\n");
    
    // Start CPU timer
    clock_t cpu_start = clock();
    

    vectorMulCPU(h_A_linear, h_B_linear, h_C_linear, numElements);
    printLinearMatrix(h_C_linear, numElements);

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
    CUDA_CHECK(cudaMemcpy(d_A, h_A_linear, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_linear, size, cudaMemcpyHostToDevice));
    
    // Set up execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    
    // // Create timing events
    cudaEvent_t start, stop;
    startTimer(&start, &stop);
    
    // // Launch kernel
    // take out rows and cols // then run them in parallel.
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // // Record execution time
    float milliseconds = stopTimer(start, stop);
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_linear, d_C, size, cudaMemcpyDeviceToHost));    

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    

    //
    // First free each row
    for (int i = 0; i < numElements; i++) {
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);
    }

    // Then free the array of pointers
    free(h_A);
    free(h_B);
    free(h_C);

    // Free host memory
    free(h_A_linear);
    free(h_B_linear);
    free(h_C_linear);
    
    printf("Done\n");
    return 0;
}

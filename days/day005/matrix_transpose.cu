#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <string>
#include <fstream>
#include "data_utils.cuh"


// CUDA kernel 
__global__ void matrixTranspose(const float *A, float *C, 
                                int numElements, int width) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if ((i < width) && (j<width)) {
        C[i*width+j]=A[width*j+i];
    }
}

// CPU implementation 
void matrixTranspose_CPU( const float *A, float *C, 
                          int numElements, int width) {
    // definition of transpose is
    // rows -> cols
    // cols -> rows
    // for row 
    // pick first elements of the columns
    for (int i = 0; i < width; i++) {
        // printf("Multiplyin %0.3f with %0.3f = %0.3f \n", A[i], C[i]);
        // [1,2,3,4] -> [1,3,2,4]
        // -----
        // [1,2]
        // [3,4]
        // row of a matrix accessed as [i]
        // pick out element of the cols
        for (int cols=0; cols<width; cols++) {
            C[i*width+cols]=A[width*cols+i];
            // printf("Marking old array index[%d] with value [%0.3f] to new index [%d] to a new value [%0.3f] \n"
            //          , width*cols+i, A[width*cols+i], i*width+cols, C[i*width+cols]
            //         );
        }

    }
}

// Measure CPU performance
double benchmarkCPU(const float *A, float *C, 
                    int numElements, int iterations, int width
            ) {

    clock_t start = clock();
    
    for (int iter = 0; iter < iterations; iter++) {
        matrixTranspose_CPU(A, C, numElements, width);
    }
    
    clock_t end = clock();
    return 1000.0 * (end - start) / CLOCKS_PER_SEC / iterations;
}

// Measure GPU performance
double benchmarkGPU(const float *h_A, float *h_C, int numElements, 
                    int iterations, int width) {
    size_t size = numElements * sizeof(float);
    
    // Allocate device memory
    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    
    // Set kernel configuration
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    // printf("Warmup! \n");
    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements, width);
    cudaDeviceSynchronize();
    
    float totalTime = 0.0f;
    
    // Benchmark iterations
    // printf("Real work! \n");
    cudaEventRecord(start);
    for (int iter = 0; iter < iterations; iter++) {
        matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements, width);
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
    std::ofstream resultsFile("benchmark_results_mat_transpose.csv");
    resultsFile << "MatrixSize,TotalElements,CPU_Time_ms,GPU_Time_ms,Speedup\n";
    
    // Test different matrix sizes
    std::vector<int> matrixSizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};

    int iterations = 10; // Number of iterations for each test
    
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
                                      iterations, size);
        printf("  CPU time: %.3f ms\n", cpuTime);


        // printLinearMatrix(h_C_cpu, size);

        
        // Run GPU benchmark
        double gpuTime = benchmarkGPU(h_A, h_C_gpu, totalElements, 
                                      iterations, size);
        printf("  GPU time: %.3f ms\n", gpuTime);

        // printLinearMatrix(h_C_gpu, size);

        
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
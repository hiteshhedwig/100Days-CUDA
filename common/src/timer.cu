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

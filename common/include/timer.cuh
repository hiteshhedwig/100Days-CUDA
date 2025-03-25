#ifndef TIMER_CUH
#define TIMER_CUH

#include <cuda_runtime.h>

// Start CUDA timer
void startTimer(cudaEvent_t *start, cudaEvent_t *stop);

// Stop CUDA timer and return elapsed time in milliseconds
float stopTimer(cudaEvent_t start, cudaEvent_t stop);

#endif // TIMER_CUH

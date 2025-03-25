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

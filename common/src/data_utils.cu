#include "data_utils.cuh"
#include <stdlib.h>
#include <math.h>

void initializeArray(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() / (float)RAND_MAX;
    }
}

void initializeMatrix(float **mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mat[i][j] = rand() / (float)RAND_MAX; // Access element directly
        }
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

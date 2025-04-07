#include "data_utils.cuh"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>


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

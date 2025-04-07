#ifndef DATA_UTILS_CUH
#define DATA_UTILS_CUH

// Initialize array with random values
void initializeArray(float *arr, int size);
void initializeMatrix(float **mat, int size) ;

// Verify results of vector addition
bool verifyVectorAdd(const float *A, const float *B, const float *C, int size);
void printLinearMatrix(float *linearMatrix, int size) ;
void printMatrix(float **matrix, int size) ;

#endif // DATA_UTILS_CUH

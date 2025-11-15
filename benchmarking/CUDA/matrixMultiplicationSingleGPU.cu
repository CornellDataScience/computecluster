#include <cstdio>   
#include <cstdlib>      
#include <cmath>           
#include <cuda_runtime.h>  

#define CHECK_CUDA(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)


__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int K, int N) {

    // row anc column this thread will be responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check (threads outside matrix do nothing)
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    // Dot product of row of A and column of B
    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];   // A[row, k]
        float b = B[k * N + col];   // B[k, col]
        sum += a * b;
    }

    C[row * N + col] = sum;
}
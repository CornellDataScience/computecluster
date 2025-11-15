#include <cstdio>   
#include <cstdlib>      
#include <cmath>           
#include <cuda_runtime.h>  
#include <chrono>  

#define CHECK_CUDA(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

void initRandom(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool verifyResult(const float* A, const float* B, const float* C,
                  int M, int K, int N) {
    //Compare gpu results with the cpu ones
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            float diff = std::fabs(sum - C[row * N + col]);
            if (diff > 1e-3f) {
                printf("Mismatch at (%d, %d): GPU=%f, CPU=%f, diff=%f\n",
                       row, col, C[row * N + col], sum, diff);
                return false;
            }
        }
    }
    return true;
}

__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int K, int N) {

    // row anc column this thread will be responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    // Dot product of row of A and column of B
    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];  
        float b = B[k * N + col];   
        sum += a * b;
    }

    C[row * N + col] = sum;
}


int main() {
    // Matrix dimensions
    int M = 2048; 
    int K = 2048; 
    int N = 2048; 

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    //assign random vals
    srand(0);
    initRandom(h_A, M * K);
    initRandom(h_B, K * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeC));

    // Copy A and B to gpu
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // how many threads we want to use on gpu
    dim3 blockDim(32, 32);  
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    printf("Launching kernel with grid=(%d,%d), block=(%d,%d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Launch kernel
    matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CHECK_CUDA(cudaGetLastError()); 

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float gpuMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpuMs, start, stop));
    printf("GPU matmul time: %.3f ms\n", gpuMs);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    auto cpuStart = std::chrono::high_resolution_clock::now();
    bool ok = verifyResult(h_A, h_B, h_C, M, K, N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    printf("CPU (verifyResult) time: %.3f ms\n", cpuMs);

    printf("Verification: %s\n", ok ? "SUCCESS" : "FAILURE");

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

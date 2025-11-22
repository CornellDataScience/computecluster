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
    int M = 8192; 
    int K = 8192; 
    int N = 8192; 

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        fprintf(stderr, "Need at least 2 GPUs, found %d\n", deviceCount);
        return EXIT_FAILURE;
    }

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

    // Split work between two GPUs by rows of A/C
    int M0 = M / 2;        // rows handled by GPU 0
    int M1 = M - M0;       // rows handled by GPU 1

    size_t sizeA0 = (size_t)M0 * K * sizeof(float);
    size_t sizeC0 = (size_t)M0 * N * sizeof(float);

    size_t sizeA1 = (size_t)M1 * K * sizeof(float);
    size_t sizeC1 = (size_t)M1 * N * sizeof(float);

    // how many threads we want to use on gpu
    dim3 blockDim(32, 32);  

    dim3 gridDim0((N + blockDim.x - 1) / blockDim.x,
                  (M0 + blockDim.y - 1) / blockDim.y);

    dim3 gridDim1((N + blockDim.x - 1) / blockDim.x,
                  (M1 + blockDim.y - 1) / blockDim.y);

    CHECK_CUDA(cudaSetDevice(0));
    float *d_A0, *d_B0, *d_C0;
    CHECK_CUDA(cudaMalloc((void**)&d_A0, sizeA0));
    CHECK_CUDA(cudaMalloc((void**)&d_B0, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C0, sizeC0));

    // Copy first half of A and full B
    CHECK_CUDA(cudaMemcpy(d_A0, h_A, sizeA0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B0, h_B, sizeB, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaSetDevice(1));
    float *d_A1, *d_B1, *d_C1;
    CHECK_CUDA(cudaMalloc((void**)&d_A1, sizeA1));
    CHECK_CUDA(cudaMalloc((void**)&d_B1, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C1, sizeC1));

    // Copy second half of A (starting at row M0) and full B
    CHECK_CUDA(cudaMemcpy(d_A1, h_A + (size_t)M0 * K, sizeA1, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B1, h_B, sizeB, cudaMemcpyHostToDevice));

    auto gpuStart = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaSetDevice(0));
    matmulNaive<<<gridDim0, blockDim>>>(d_A0, d_B0, d_C0, M0, K, N);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaSetDevice(1));
    matmulNaive<<<gridDim1, blockDim>>>(d_A1, d_B1, d_C1, M1, K, N);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto gpuEnd = std::chrono::high_resolution_clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
    printf("2-GPU matmul time: %.3f ms\n", gpuMs);

    // First half of C from GPU 0
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemcpy(h_C, d_C0, sizeC0, cudaMemcpyDeviceToHost));

    // Second half of C from GPU 1
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemcpy(h_C + (size_t)M0 * N, d_C1, sizeC1, cudaMemcpyDeviceToHost));

    // verification
    // auto cpuStart = std::chrono::high_resolution_clock::now();
    // bool ok = verifyResult(h_A, h_B, h_C, M, K, N);
    bool ok = true;
    // auto cpuEnd = std::chrono::high_resolution_clock::now();
    // double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    // printf("CPU (verifyResult) time: %.3f ms\n", cpuMs);

    // printf("Verification: %s\n", ok ? "SUCCESS" : "FAILURE");

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(d_A0));
    CHECK_CUDA(cudaFree(d_B0));
    CHECK_CUDA(cudaFree(d_C0));

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaFree(d_A1));
    CHECK_CUDA(cudaFree(d_B1));
    CHECK_CUDA(cudaFree(d_C1));

    free(h_A);
    free(h_B);
    free(h_C);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

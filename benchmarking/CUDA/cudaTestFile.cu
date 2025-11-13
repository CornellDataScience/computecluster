#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addKernel(int a, int b, int *c) {
    *c = a + b;
}

int main() {
    int a = 2, b = 7;
    int result = 0;
    int *d_result;

    // Allocating memory on the (single) GPU
    cudaError_t err = cudaMalloc((void**)&d_result, sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel: 1 block, 1 thread
    addKernel<<<1, 1>>>(a, b, d_result);

    // Check for any launch errors and print them
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_result);
        return 1;
    }

    // Copy all the result back to CPU
    err = cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_result);
        return 1;
    }

    cudaFree(d_result);

    printf("Result from GPU: %d + %d = %d\n", a, b, result);
    return 0;
}

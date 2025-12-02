/**
 * add.cu
 *
 * CUDA program to add two large arrays
 * Demonstrates memory management and performance impact of different block sizes
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 16384 * 16384

// convenience macro to check if CUDA calls succeed
#define CUDA_CHECK(call) {                                                          \
   const cudaError_t error = call;                                                  \
   if (error != cudaSuccess) {                                                      \
       fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                       \
       fprintf(stderr, "code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
       exit(1);                                                                     \
   }                                                                                \
}

// addKernel: computes element-wise addition of two arrays at index idx
__global__ void addKernel(float *c, float *a, float *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // allocate host memory for N floats
    size_t bytes = N * sizeof(float);
    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *c = (float *)malloc(bytes);

    // initialise a and b with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    // allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // copy input arrays from host to device
    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    // set up different block sizes for timing
    int threadsPerBlock[] = {64, 128, 256, 512, 1024, 800};
    int numRuns = sizeof(threadsPerBlock) / sizeof(threadsPerBlock[0]);

    // warm up the cache
    addKernel<<<(N + 256 - 1) / 256, 256>>>(d_c, d_a, d_b);
    CUDA_CHECK(cudaDeviceSynchronize());

    // launch kernels with different block sizes
    for (int i = 0; i < numRuns; i++) {
        cudaMemset(d_c, 0, bytes);

        int blockSize = threadsPerBlock[i];
        // round up to multiple of blockSize
        int numBlocks = (N + blockSize - 1) / blockSize;

        clock_t start = clock(), end;

        // launch the kernel
        addKernel<<<numBlocks, blockSize>>>(d_c, d_a, d_b);

        // wait for GPU to finish
        CUDA_CHECK(cudaDeviceSynchronize());

        // measure time taken for kernel execution
        end = clock();
        double ktime = (double)(end - start) / CLOCKS_PER_SEC;

        // copy result back to host
        CUDA_CHECK(cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

        // measure total time taken, including memcpy
        end = clock();
        double etime = (double)(end - start) / CLOCKS_PER_SEC;

        // print timing results
        printf("addKernel<<<%d,%d>>>\t Kernel: %f seconds\tTotal: %f seconds\n",
            numBlocks, blockSize, ktime, etime);
    }

    // free memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(a);
    free(b);
    free(c);
}
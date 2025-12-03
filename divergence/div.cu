/**
 * div.cu
 *
 * Demonstrates the performance impact of warp divergence
 * Each divKernelN function introduces divergence with N different branches within a warp
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define N 16384 * 16384
#define NUM_ITERS 100
#define BLOCK_SIZE 256

// work: simulate work on the device
__device__ __forceinline__ float work(float x) {
    #pragma unroll 1 // prevent loop unrolling
    for (int i = 0; i < NUM_ITERS; i++) {
        x = sinf(x) * cosf(x);
    }
    return x;
}

/**
 * divKernelN: warp divergence with N different branches
 * For larger N, more divergence occurs, so more performance degradation is expected
 */

__global__ void divKernel1(float *out, const float *in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[idx];
    float result = work(x);
    out[idx] = result;
}

__global__ void divKernel2(float *out, const float *in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31; // lane ID within warp

    float x = in[idx];
    float result;

    if (lane % 2) {
        result = work(x);
    } else {
        result = work(x + 1);
    }

    out[idx] = result;
}

__global__ void divKernel8(float *out, const float *in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31; // lane ID within warp

    float x = in[idx];
    float result;

    if (lane < 4) {
        result = work(x);
    } else if (lane < 8) {
        result = work(x + 1);
    } else if (lane < 12) {
        result = work(x + 2);
    } else if (lane < 16) {
        result = work(x + 3);
    } else if (lane < 20) {
        result = work(x + 4);
    } else if (lane < 24) {
        result = work(x + 5);
    } else if (lane < 28) {
        result = work(x + 6);
    } else {
        result = work(x + 7);
    }

    out[idx] = result;
}

__global__ void divKernel16(float *out, const float *in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31; // lane ID within warp

    float x = in[idx];
    float result;

    if (lane < 2) {
        result = work(x);
    } else if (lane < 4) {
        result = work(x + 1);
    } else if (lane < 6) {
        result = work(x + 2);
    } else if (lane < 8) {
        result = work(x + 3);
    } else if (lane < 10) {
        result = work(x + 4);
    } else if (lane < 12) {
        result = work(x + 5);
    } else if (lane < 14) {
        result = work(x + 6);
    } else if (lane < 16) {
        result = work(x + 7);
    } else if (lane < 18) {
        result = work(x + 8);
    } else if (lane < 20) {
        result = work(x + 9);
    } else if (lane < 22) {
        result = work(x + 10);
    } else if (lane < 24) {
        result = work(x + 11);
    } else if (lane < 26) {
        result = work(x + 12);
    } else if (lane < 28) {
        result = work(x + 13);
    } else if (lane < 30) {
        result = work(x + 14);
    } else {
        result = work(x + 15);
    }

    out[idx] = result;
}

__global__ void divKernel32(float *out, const float *in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31; // lane ID within warp

    float x = in[idx];
    float result;

    switch (lane) {
    case 0: result = work(x); break;
    case 1: result = work(x + 1); break;
    case 2: result = work(x + 2); break;
    case 3: result = work(x + 3); break;
    case 4: result = work(x + 4); break;
    case 5: result = work(x + 5); break;
    case 6: result = work(x + 6); break;
    case 7: result = work(x + 7); break;
    case 8: result = work(x + 8); break;
    case 9: result = work(x + 9); break;
    case 10: result = work(x + 10); break;
    case 11: result = work(x + 11); break;
    case 12: result = work(x + 12); break;
    case 13: result = work(x + 13); break;
    case 14: result = work(x + 14); break;
    case 15: result = work(x + 15); break;
    case 16: result = work(x + 16); break;
    case 17: result = work(x + 17); break;
    case 18: result = work(x + 18); break;
    case 19: result = work(x + 19); break;
    case 20: result = work(x + 20); break;
    case 21: result = work(x + 21); break;
    case 22: result = work(x + 22); break;
    case 23: result = work(x + 23); break;
    case 24: result = work(x + 24); break;
    case 25: result = work(x + 25); break;
    case 26: result = work(x + 26); break;
    case 27: result = work(x + 27); break;
    case 28: result = work(x + 28); break;
    case 29: result = work(x + 29); break;
    case 30: result = work(x + 30); break;
    case 31: result = work(x + 31); break;
    }

    out[idx] = result;
}

// divKernel32Improved: an improved version of divKernel32 that reduces divergence
//                      by using branchless programming
__global__ void divKernel32Improved(float *out, const float *in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31; // lane ID within warp

    float x = in[idx];
    float result;

    // implemented during lecture
    result = work(x + lane);

    out[idx] = result;
}

int main() {
    // allocate host memory for N floats
    // for demo purposes, we only need input array, so no need to allocate output array
    size_t bytes = N * sizeof(float);
    float *in = (float *)malloc(bytes);

    // initialise in with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        in[i] = (float)rand() / RAND_MAX;
    }

    // allocate device memory
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);

    // copy input array from host to device
    cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

    // warm up the cache
    divKernel1<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();

    clock_t start, end;
    double ktime;

    start = clock();
    divKernel1<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();
    // ordinarily, memcpy from device to host here, but we skip it to focus on kernel time
    end = clock();
    ktime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Kernel Time with 1 branch:\t\t\t%f seconds\n", ktime);

    start = clock();
    divKernel2<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();
    end = clock();
    ktime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Kernel Time with 2 branches:\t\t\t%f seconds\n", ktime);

    start = clock();
    divKernel8<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();
    end = clock();
    ktime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Kernel Time with 8 branches:\t\t\t%f seconds\n", ktime);

    start = clock();
    divKernel16<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();
    end = clock();
    ktime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Kernel Time with 16 branches:\t\t\t%f seconds\n", ktime);

    start = clock();
    divKernel32<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();
    end = clock();
    ktime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Kernel Time with 32 branches:\t\t\t%f seconds\n", ktime);

    start = clock();
    divKernel32Improved<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();
    end = clock();
    ktime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Kernel Time with 32 branches (improved):\t%f seconds\n", ktime);

    // free memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(in);
}
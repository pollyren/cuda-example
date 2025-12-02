/**
 * hello32.cu
 *
 * Hello, world, but with 32 threads on the GPU
 *
 */

#include <stdio.h>

__global__ void helloKernel() {
    printf("Hello, world from GPU!\n");
}

int main() {
    helloKernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}

/**
 * hello.cu
 *
 * Our first GPU program :)
 *
 */

#include <stdio.h>

__global__ void helloKernel() {
    printf("Hello, world from GPU!\n");
}

int main() {
    helloKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

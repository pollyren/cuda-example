/**
 * add.c
 *
 * CPU program to add two large arrays
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 16384 * 16384

// add: c = a + b
void add(float *c, float *a, float *b) {
    *c = *a + *b;
}

int main() {
    // allocate enough memory for N floats
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

    // warm up the cache
    for (int i = 0; i < N; i++) {
        add(&c[i], &a[i], &b[i]);
    }

    // perform addition with timing
    clock_t start = clock();
    for (int i = 0; i < N; i++) {
        add(&c[i], &a[i], &b[i]);
    }
    clock_t end = clock();
    double etime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU: %f seconds\n", etime);

    free(a);
    free(b);
    free(c);
}
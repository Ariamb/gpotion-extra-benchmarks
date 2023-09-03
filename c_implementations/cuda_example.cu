#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    if (tid < 10) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int a[10], b[10], c[10];
    int *d_a, *d_b, *d_c;

    for (int i = 0; i < 10; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMalloc((void **)&d_a, 10 * sizeof(int));
    cudaMalloc((void **)&d_b, 10 * sizeof(int));
    cudaMalloc((void **)&d_c, 10 * sizeof(int));

    cudaMemcpy(d_a, a, 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 10 * sizeof(int), cudaMemcpyHostToDevice);

    add<<<10, 1>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

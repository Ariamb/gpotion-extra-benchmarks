#include <stdio.h>
#include <time.h>

#define imin(a,b) (a<b?a:b)

const int threadsPerBlock = 256;


void generateLog(double time, int iteration, int n){
  printf("Writting operation time to file.\n");
  FILE *file;
  char filename[] = "time-c-cpudotproduct.txt";

  file = fopen(filename, "a");
  fprintf(file, "time: %f \t iteration: %d \t array size: %d \n", time, iteration, n);
  fclose(file);
  
  printf("done.\n");


}


__global__ void dot(float* a, float* b, float* c, int N) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while (tid < N){
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	// set the cache values
	cache[cacheIndex] = temp;

	// synchronize threads in this block
	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0){
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];

		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

int main (int argc, char *argv[]) {
	clock_t start_time, end_time;
    double cpu_time_used;
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	
	int N = atoi(argv[1]);
    int iteration = atoi(argv[2]);

	int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);

	// allocate memory on the cpu side
	a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));
	//sqr = (float*)malloc(N*sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid*sizeof(float));

	for(int i=0; i<N; i++) {
		a[i] = i;
		b[i] = i;
	}

	// allocate the memory on the gpu
		start_time = clock();

	cudaMalloc((void**)&dev_a, N*sizeof(float));
	cudaMalloc((void**)&dev_b, N*sizeof(float));
	cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float));

	// copy the arrays 'a' and 'b' to the gpu
	cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

	dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c, N);

	// copy the array 'c' back from the gpu to the cpu
	cudaMemcpy(partial_c,dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

	// finish up on the cpu side
	c = 0;
	for(int i=0; i<blocksPerGrid; i++) {
		c += partial_c[i];
	}

	// free memory on the gpu side
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
    end_time = clock();

	// free memory on the cpu side
	// valor real, real, tri real da sum of squares 12861782365696
	free(a);
	free(b);
	free(partial_c);
    cpu_time_used = ((double) (end_time - start_time) * 1000000) / CLOCKS_PER_SEC;
	generateLog(cpu_time_used, iteration, N);


}
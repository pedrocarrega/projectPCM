#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <sys/time.h>
//#include <Windows.h>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define GRAPH_SIZE 6144
#define WORK_SIZE 256
#define NTHREADS 1024
#define BLOCKS 16

#define EDGE_COST(graph, GRAPH_SIZE, a, b) graph[a * GRAPH_SIZE + b]
#define D(a, b) EDGE_COST(output, GRAPH_SIZE, a, b)

#define INF 0x1fffffff


//createGraph
void generate_random_graph(int* output) {
	int i, j;

	srand(0xdadadada);

	for (i = 0; i < GRAPH_SIZE; i++) {
		for (j = 0; j < GRAPH_SIZE; j++) {
			if (i == j) {
				D(i, j) = 0;
			}
			else {
				int r;
				r = rand() % 40;
				if (r > 20) {
					r = INF;
				}

				D(i, j) = r;
			}
		}
	}
}

//sequencial GPU
__global__ void calculateSequencialGPU(int* output) {

	int i, j, k;

	for (k = 0; k < GRAPH_SIZE; k++) {
		for (i = 0; i < GRAPH_SIZE; i++) {
			for (j = 0; j < GRAPH_SIZE; j++) {
				if (D(i, k) + D(k, j) < D(i, j)) {
					D(i, j) = D(i, k) + D(k, j);
				}
			}
		}
	}
}

__global__ void calcWithoutAtomic1D(int* output, int k) {

	int totalID = blockIdx.x * blockDim.x * WORK_SIZE + threadIdx.x * WORK_SIZE;
	int i = totalID / GRAPH_SIZE;
	int j = totalID % GRAPH_SIZE;

	int counter = 0;
	while (counter < WORK_SIZE)
	{
		if (D(i, k) + D(k, j) < D(i, j)) {
			D(i, j) = D(i, k) + D(k, j);
		}
		if ((j + 1) < GRAPH_SIZE) {
			j++;
		}else {
			i++;
			j = 0;
		}
		counter++;
	}
}

__global__ void calcWithAtomic1D(int* output, int* syncGrid) {

	int totalID = blockIdx.x * blockDim.x * WORK_SIZE + threadIdx.x * WORK_SIZE;
	int i, j, counter, k = 0, avaliador = 0, Dik;

	while(k < GRAPH_SIZE){
		i = totalID / GRAPH_SIZE;
		j = totalID % GRAPH_SIZE;
		counter = 0;
		syncGrid[blockIdx.x] = 0;
		while (counter < WORK_SIZE)
		{
			if (D(i,k) + D(k, j) < D(i, j)) {
				D(i, j) = D(i,k) + D(k, j);
			}
			if (j + 1 < GRAPH_SIZE) {
				j++;
			}else {
				i++;
				j = 0;
			}
			counter++;
		}
		k++;

		if (threadIdx.x == 0) {
			syncGrid[blockIdx.x] = 1;
			do{
				avaliador = 0;
				for(int i = 0; i < gridDim.x; i++){
					avaliador += syncGrid[i];
				}
			}while(avaliador != gridDim.x);
		
		}
		__syncthreads();
	}
}

__global__ void calcWithAtomic1DMem(int* output, int* syncGrid) {
	
	int totalID = blockIdx.x * blockDim.x * WORK_SIZE + threadIdx.x * WORK_SIZE;
	int i, j, counter, k = 0, avaliador = 0, Dik;

	while(k < GRAPH_SIZE){
		i = totalID / GRAPH_SIZE;
		j = totalID % GRAPH_SIZE;
		counter = 0;
		Dik = D(i,k);
		syncGrid[blockIdx.x] = 0;
		while (counter < WORK_SIZE)
		{
			if (Dik + D(k, j) < D(i, j)) {
				D(i, j) = Dik + D(k, j);
			}
			if (j + 1 < GRAPH_SIZE) {
				j++;
			}else {
				i += ((i+1)< GRAPH_SIZE);
				j = 0;
				Dik = D(i,k);
			}
			counter++;
		}
		k++;

		if (threadIdx.x == 0) {
			syncGrid[blockIdx.x] = 1;
			do{
				avaliador = 0;
				for(int i = 0; i < gridDim.x; i++){
					avaliador += syncGrid[i];
				}
			}while(avaliador != gridDim.x);
		
		}
		__syncthreads();
	}
}

void floyd_warshall_gpu(const int* graph, int* output) {

	int* dev_a;
	cudaMalloc(&dev_a, sizeof(int) * GRAPH_SIZE * GRAPH_SIZE);
	cudaMemcpy(dev_a, graph, sizeof(int) * GRAPH_SIZE * GRAPH_SIZE, cudaMemcpyHostToDevice);

	//calculateSequencialGPU << <1, 1 >> > (dev_a, GRAPH_SIZE);

	int blocks;
	int threads;
	cudaOccupancyMaxPotentialBlockSize (&blocks, &threads, calcWithoutAtomic1D, 0, GRAPH_SIZE*GRAPH_SIZE);
	//blocks = sqrt(blocks);
	//threads = sqrt(threads);
	int workPerThread = ((GRAPH_SIZE*GRAPH_SIZE) / (threads*blocks));// + 1;
	printf("workPerThread= %d, blocks= %d threadsPerBlocks = %d\n", workPerThread, blocks, threads);

	int* syncGrid;
	cudaMalloc(&syncGrid, sizeof(int) * blocks);
	
	for (int k = 0; k < GRAPH_SIZE; k++) {
		calcWithoutAtomic1D <<<blocks, threads>>> (dev_a, k);
	}
	
	//calcWithAtomic1D<<<blocks, threads>>> (dev_a, syncGrid);
	//calcWithAtomic1DShared<<<blocks, threads>>> (dev_a, syncGrid);

	cudaError_t err = cudaMemcpy(output, dev_a, sizeof(int) * GRAPH_SIZE * GRAPH_SIZE, cudaMemcpyDeviceToHost);
	gpuErrchk(err);
	cudaFree(dev_a);
	cudaFree(syncGrid);//teste
}

void floyd_warshall_cpu(const int* graph, int* output) {
	int i, j, k;

	for (k = 0; k < GRAPH_SIZE; k++) {
		for (i = 0; i < GRAPH_SIZE; i++) {
			for (j = 0; j < GRAPH_SIZE; j++) {
				if (D(i, k) + D(k, j) < D(i, j)) {
					D(i, j) = D(i, k) + D(k, j);
				}
			}
		}
	}
}

int main(int argc, char** argv) {
/*
	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	double interval;*/


	#define TIMER_START() gettimeofday(&tv1, NULL)
	#define TIMER_STOP()                                                           \
  		gettimeofday(&tv2, NULL);                                                    \
  		timersub(&tv2, &tv1, &tv);                                                   \
  		time_delta = (float)tv.tv_sec + tv.tv_usec / 1000000.0

  	struct timeval tv1, tv2, tv;
  	float time_delta;	

  	int* graph, * output_cpu, * output_gpu;
	int size;

	size = sizeof(int) * GRAPH_SIZE * GRAPH_SIZE;

	graph = (int*)malloc(size);
	assert(graph);

	output_cpu = (int*)malloc(size);
	assert(output_cpu);
	memset(output_cpu, 0, size);

	output_gpu = (int*)malloc(size);
	assert(output_gpu);

	generate_random_graph(graph);

	fprintf(stderr, "running on cpu...\n");
	TIMER_START();
	memcpy(output_cpu, graph, sizeof(int) * GRAPH_SIZE * GRAPH_SIZE);

	//QueryPerformanceFrequency(&frequency);
	//QueryPerformanceCounter(&start);
	floyd_warshall_cpu(graph, output_cpu);
	TIMER_STOP();
	fprintf(stderr, "%f seconds\n", time_delta);

	//QueryPerformanceCounter(&end);
	//interval = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	//fprintf(stderr, "%f seconds\n", interval);

	fprintf(stderr, "running on gpu...\n");
	TIMER_START();
	//QueryPerformanceFrequency(&frequency);
	//QueryPerformanceCounter(&start);
	floyd_warshall_gpu(graph, output_gpu);
	TIMER_STOP();
	fprintf(stderr, "%f seconds\n", time_delta);

	//QueryPerformanceCounter(&end);
	//interval = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	//fprintf(stderr, "%f seconds\n", interval);



	if (memcmp(output_cpu, output_gpu, size) != 0) {
		fprintf(stderr, "FAIL!\n");
	}
	else {
		fprintf(stderr, "Verified!\n");
	}
	
	return 0;
}

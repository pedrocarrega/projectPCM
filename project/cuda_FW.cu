#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <Windows.h>
//for __syncthreads()
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

#define GRAPH_SIZE 1000

#define EDGE_COST(graph, graph_size, a, b) graph[a * graph_size + b]
#define D(a, b) EDGE_COST(output, graph_size, a, b)

#define INF 0x1fffffff

void generate_random_graph(int* output, int graph_size) {
	int i, j;

	srand(0xdadadada);

	for (i = 0; i < graph_size; i++) {
		for (j = 0; j < graph_size; j++) {
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

__global__ void calcOnePosPerThread(int* output, int graph_size, int k)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x);
	int j = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i < graph_size && j < graph_size) {
		if (D(i, k) + D(k, j) < D(i, j)) {
			D(i, j) = D(i, k) + D(k, j);
		}
	}
}

__global__ void calThreadPerColumn(int* output, const int graph_size, int numThreads, int k)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int localStorageRegister;


	while (i < graph_size) {

		localStorageRegister = D(i, k);

		for (int j = 0; j < graph_size; j++)
		{
			if (localStorageRegister + D(k, j) < D(i, j)) {
				D(i, j) = localStorageRegister + D(k, j);
			}
		}
		i = i + numThreads;
	}
}

__device__ int barrier = 0;

__global__ void calcWithAtomic(int* output, int graph_size, int workPerThread, int numBlocks)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * workPerThread;
	int j = (blockIdx.y * blockDim.y + threadIdx.y) * workPerThread;
	int k = 0;

	while (k < graph_size) {
		for (int x = i; x < i + workPerThread; x++)
		{
			for (int y = j; y < j + workPerThread; y++)
			{
				if (x < graph_size && y < graph_size) {
					if (D(x, k) + D(k, y) < D(x, y)) {
						D(x, y) = D(x, k) + D(k, y);
					}
				}
			}
		}

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			atomicAdd(&barrier, 1);
		}

		k = k + 1;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			while ((atomicCAS(&barrier, numBlocks, numBlocks) % numBlocks) != 0);
		}
		__syncthreads();

	}
}

__global__ void calcWithoutAtomic(int* output, int graph_size, int k, int workPerThread)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * workPerThread;
	int j = (blockIdx.y * blockDim.y + threadIdx.y) * workPerThread;

	for (int x = i; x < i + workPerThread; x++)
	{
		for (int y = j; y < j + workPerThread; y++)
		{
			if (x < graph_size && y < graph_size) {
				if (D(x, k) + D(k, y) < D(x, y)) {
					D(x, y) = D(x, k) + D(k, y);
				}
			}
		}
	}
}

__global__ void calculateSequencialGPU(int* output, int graph_size)
{
	int i, j, k;

	for (k = 0; k < graph_size; k++) {
		for (i = 0; i < graph_size; i++) {
			for (j = 0; j < graph_size; j++) {
				if (D(i, k) + D(k, j) < D(i, j)) {
					D(i, j) = D(i, k) + D(k, j);
				}
			}
		}
	}
}

__global__ void calculateUsingSharedMemoryGPU(int* output, int graph_size)
{

}

void floyd_warshall_gpu(const int* graph, int graph_size, int* output) {

	int* dev_a;
	cudaMalloc(&dev_a, sizeof(int) * graph_size * graph_size);
	cudaMemcpy(dev_a, graph, sizeof(int) * graph_size * graph_size, cudaMemcpyHostToDevice);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int NThreads = 8;
	//for safety
	if (NThreads > 32) {
		NThreads = sqrt(prop.maxThreadsPerBlock);
	}

	int maxBlocksPerAxis = sqrt(prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / (NThreads * NThreads)));
	int maxThreadsPerAxis = maxBlocksPerAxis * NThreads;
	int workPerThread = ((graph_size) / maxThreadsPerAxis) + 1;

	dim3 threads(NThreads, NThreads);
	dim3 blocks(maxBlocksPerAxis, maxBlocksPerAxis);

	int t = 64;
	int b = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / t);

	for (int k = 0; k < graph_size; k++) {
		//calcWithoutAtomic << <blocks, threads >> > (dev_a, graph_size, k, workPerThread);
		//calcOnePosPerThread<<<dim3(GRAPH_SIZE/NThreads,GRAPH_SIZE/NThreads), dim3(NThreads,NThreads)>>>(dev_a, graph_size,k);
		calThreadPerColumn <<<b, t >>> (dev_a, graph_size, t * b, k);
	}
	//calcWithAtomic << <blocks, threads >> > (dev_a, graph_size, workPerThread, maxBlocksPerAxis*maxBlocksPerAxis);

	cudaError_t err = cudaMemcpy(output, dev_a, sizeof(int) * graph_size * graph_size, cudaMemcpyDeviceToHost);
	gpuErrchk(err);
	cudaFree(dev_a);
}

void floyd_warshall_cpu(const int* graph, int graph_size, int* output) {
	int i, j, k;

	memcpy(output, graph, sizeof(int) * graph_size * graph_size);

	for (k = 0; k < graph_size; k++) {
		for (i = 0; i < graph_size; i++) {
			for (j = 0; j < graph_size; j++) {
				if (D(i, k) + D(k, j) < D(i, j)) {
					D(i, j) = D(i, k) + D(k, j);
				}
			}
		}
	}
}

int main(int argc, char** argv) {
	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	double interval;

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

	generate_random_graph(graph, GRAPH_SIZE);

	fprintf(stderr, "running on cpu...\n");
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);
	floyd_warshall_cpu(graph, GRAPH_SIZE, output_cpu);
	QueryPerformanceCounter(&end);
	interval = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	fprintf(stderr, "%f seconds\n", interval);

	fprintf(stderr, "running on gpu...\n");
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);
	floyd_warshall_gpu(graph, GRAPH_SIZE, output_gpu);
	QueryPerformanceCounter(&end);
	interval = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	fprintf(stderr, "%f seconds\n", interval);

	if (memcmp(output_cpu, output_gpu, size) != 0) {
		fprintf(stderr, "FAIL!\n");
	}

	return 0;
}
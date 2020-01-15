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

#define GRAPH_SIZE 2048
#define WORK_SIZE 96
#define NTHREADS 32
#define BLOCKS 3

#define EDGE_COST(graph, graph_size, a, b) graph[a * graph_size + b]
#define D(a, b) EDGE_COST(output, graph_size, a, b)

#define INF 0x1fffffff


//createGraph
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

//calcOnePositionPerThread e deixar o schedualing para a gpu, fazendo assim todas as posicoes da matriz
__global__ void calcOnePosPerThread(int* output, int graph_size, int k)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x);
	int j = (blockIdx.y * blockDim.y + threadIdx.y);

	//while (i < graph_size && j < graph_size) {
		if (D(i, k) + D(k, j) < D(i, j)) {
			D(i, j) = D(i, k) + D(k, j);
		}
		//i += blockDim.x * gridDim.x;
		//j += blockDim.y * gridDim.y;
	//}
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

/*
Problemas em todos os que usam atomic possivelmente devido a estar a calcular mal o num max
de threads/blocks/warps que se pode ter na totalidade assim como por SM, ver descriao nas doubts.txt
*/
__global__ void calcWithAtomic(int* output, int graph_size)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * WORK_SIZE;
	int j = (blockIdx.y * blockDim.y + threadIdx.y) * WORK_SIZE;
	int k = 0;
	int numBlocks = gridDim.x * gridDim.y;

	while (k < graph_size) {
		if (threadIdx.x == 0 && threadIdx.y == 0)
		printf("Before\n");
		for (int x = i; x < i + WORK_SIZE; x++)
		{
			for (int y = j; y < j + WORK_SIZE; y++)
			{
				if (x < graph_size && y < graph_size) {
					if (D(x, k) + D(k, y) < D(x, y)) {
						D(x, y) = D(x, k) + D(k, y);
					}
				}else{
					break;
				}
			}
			if (threadIdx.x == 0 && threadIdx.y == 0)
			printf("AFTER: %d\n", x);
		}

		k++;
		/*if (threadIdx.x == 0 && threadIdx.y == 0)
		printf("After\n");
		*/
		//bloco perde-se
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			atomicAdd(&barrier,1);
			printf("barrier: %d\n", barrier);
			while (barrier % numBlocks != 0);
		}
		
		__syncthreads();

	}
}

__global__ void calcWithoutAtomic(int* output, int graph_size, int k, int workPerThread)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * workPerThread;
	int j = (blockIdx.y * blockDim.y + threadIdx.y) * workPerThread;

	//int xk, ky;

	for (int x = i; x < i + workPerThread; x++)
	{
		//xk = D(x, k);
		for (int y = j; y < j + workPerThread; y++)
		{
			//ky = D(k, y);
			if (x < graph_size && y < graph_size) {
				if (D(x, k) + D(k, y) < D(x, y)) {
					D(x, y) = D(x, k) + D(k, y);
				}
			}
		}
	}
}

/*
__global__ void calcSharedWithAtomic(int* output, int graph_size)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * WORK_SIZE;
	int j = (blockIdx.y * blockDim.y + threadIdx.y) * WORK_SIZE;
	int k = 0;
	int numBlocks = gridDim.x * gridDim.y;
	
	__shared__ int valuesX[NTHREADS][NTHREADS];
	__shared__ int valuesY[NTHREADS][NTHREADS][WORK_SIZE];

	while (k < graph_size) {
		for (int x = i; x < i + WORK_SIZE; x++)
		{
			if (x < graph_size) {
				valuesX[threadIdx.x][threadIdx.y] = D(x, k);
			}
			for (int y = j; y < j + WORK_SIZE; y++)
			{
				if (x < graph_size && y < graph_size) {
					if (x == i) {
						valuesY[threadIdx.x][threadIdx.y][y-j] = D(k, y);
					}
					if (valuesX[threadIdx.x][threadIdx.y] + valuesY[threadIdx.x][threadIdx.y][y - j] < D(x, y)) {
						D(x, y) = valuesX[threadIdx.x][threadIdx.y] + valuesY[threadIdx.x][threadIdx.y][y - j];
					}
				}
			}
		}

		k++;

		if (threadIdx.x == 0 && threadIdx.y) {
			atomicAdd(&barrier,1);
			while (barrier % numBlocks != 0);
		}
		__syncthreads();

	}
}


__global__ void calcSharedWithoutAtomic(int* output, int graph_size, int k, int workPerThread)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * workPerThread;
	int j = (blockIdx.y * blockDim.y + threadIdx.y) * workPerThread;
	//printf("xt = %d | yt = %d\n", threadIdx.x, threadIdx.y);
	int xT = threadIdx.x;
	int yT = threadIdx.y;

	__shared__ int valuesX[NTHREADS][NTHREADS];
	__shared__ int valuesY[NTHREADS*NTHREADS][WORK_SIZE];

	int currT = xT * blockDim.x + yT;

	//printf("workPT %d , blockdim = %d ", workPerThread, blockDim.x);
	/*
	Por enquanto o valuesX faz ser um pouco mais rapido mas o valuesY faz ficar bastante mais lento, implementar com shared memory
	*//*
	for (int x = i; x < i + workPerThread; x++)
	{
		
		if (x < graph_size) {
			valuesX[xT][yT] = D(x, k);
		}/*
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			array[blockDim.x * blockIdx.x] = D(x, k);
		}
		__syncthreads;*//*
		for (int y = j; y < j + workPerThread; y++)
		{
			//values[threadX][threadY] = D(x, k);
			//ky = D(k, y);
			if (x < graph_size && y < graph_size) {
				if (x == i) {
					valuesY[currT][y - j] = D(k, y);
				}
				if (valuesX[xT][yT] + valuesY[currT][y - j] < D(x, y)) {
					D(x, y) = valuesX[xT][yT] + valuesY[currT][y - j];
				}
			}
		}
	}
}

/*
__global__ void sharedCalcWithoutAtomic(int* output, int graph_size, int k, const int workPerThread)
{
	//size_t workSize = (workPerThread * blockDim.x)^2;

	//const int test = blockDim.x;
	
	__shared__ int array[10 * 11];
	
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * workPerThread;
	int j = (blockIdx.y * blockDim.y + threadIdx.y) * workPerThread;

	for (int x = i; x < i + workPerThread; x++)
	{
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			array[blockDim.x * blockIdx.x] = D(x, k);
		}
		__syncthreads;
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
*/


//sequencial GPU
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

void floyd_warshall_gpu(const int* graph, int graph_size, int* output) {

	int* dev_a;
	cudaMalloc(&dev_a, sizeof(int) * graph_size * graph_size);
	cudaMemcpy(dev_a, graph, sizeof(int) * graph_size * graph_size, cudaMemcpyHostToDevice);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int NThreads = NTHREADS;
	//for safety
	if (NThreads > 32) {
		NThreads = sqrt(prop.maxThreadsPerBlock);
	}

	
	//int maxMemSize = prop.sharedMemPerBlock;
	//int maxBlocksPerAxis = sqrt(prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / (NThreads * NThreads)));
	/*
	int maxThreadsPerAxis = maxBlocksPerAxis * NThreads;
	int workPerThread = ((graph_size) / maxThreadsPerAxis) + 1;

	fprintf(stderr, "work %d\nthreads %d\n", workPerThread, NThreads);
	*/

	//dim3 threads(NThreads, NThreads);
	//dim3 blocks(maxBlocksPerAxis, maxBlocksPerAxis);

	//printf("blockPerAxis = %d\n", maxBlocksPerAxis);

	//int t = 64;
	//int b = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / t);
	
	//calculateSequencialGPU << <1, 1 >> > (dev_a, graph_size);
/*
	int blocks;
	int threads;
	cudaOccupancyMaxPotentialBlockSize (&blocks, &threads, calcWithAtomic, 0, GRAPH_SIZE*GRAPH_SIZE);
	blocks = sqrt(blocks);
	threads = sqrt(threads);
	printf("workPerThread to be defined as %d\n", threads*blocks);
	
/*	
	if(threads % 2 != 0){
		threads++;
	}
*/
	
/*
	if(blocks % 2 != 0){
		blocks++;
	}*/


	
	/*
	for (int k = 0; k < graph_size; k++) {
		//calcSharedWithoutAtomic <<<blocks, threads>>> (dev_a, graph_size, k);
		//calcWithoutAtomic <<<blocks, threads>>> (dev_a, graph_size, k);
		calcOnePosPerThread <<<dim3(GRAPH_SIZE/8,GRAPH_SIZE/8), dim3(8,8)>>>(dev_a, graph_size,k);
		//calThreadPerColumn <<<b, t >>> (dev_a, graph_size, t * b, k);
	}
	*/
	
	
	
	//fprintf(stderr, "blocks: %d\nthreads: %d\n", blocks, threads);
	
	calcWithAtomic <<<dim3(BLOCKS,BLOCKS), dim3(NTHREADS,NTHREADS)>>> (dev_a, graph_size);
	//calcSharedWithAtomic <<<blocks, threads >>> (dev_a, graph_size);
	//calcSIMDSharedWithAtomic <<<blocks, threads >>> (dev_a, graph_size);

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

	generate_random_graph(graph, GRAPH_SIZE);

	fprintf(stderr, "running on cpu...\n");
	TIMER_START();
	//floyd_warshall_cpu(graph, GRAPH_SIZE, output_cpu);
	TIMER_STOP();
	fprintf(stderr, "%f seconds\n", time_delta);

	fprintf(stderr, "running on gpu...\n");
	TIMER_START();
	floyd_warshall_gpu(graph, GRAPH_SIZE, output_gpu);
	TIMER_STOP();
	fprintf(stderr, "%f seconds\n", time_delta);



	if (memcmp(output_cpu, output_gpu, size) != 0) {
		fprintf(stderr, "FAIL!\n");
	}
	else {
		fprintf(stderr, "Verified!\n");
	}

	return 0;
}

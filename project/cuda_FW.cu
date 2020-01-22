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

#define GRAPH_SIZE 2048 * 3
#define WORK_SIZE 2304
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

//calcOnePositionPerThread e deixar o schedualing para a gpu, fazendo assim todas as posicoes da matriz
__global__ void calcOnePosPerThread(int* output, int k)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x);
	int j = (blockIdx.y * blockDim.y + threadIdx.y);

	//while (i < GRAPH_SIZE && j < GRAPH_SIZE) {
		if (D(i, k) + D(k, j) < D(i, j)) {
			D(i, j) = D(i, k) + D(k, j);
		}
		//i += blockDim.x * gridDim.x;
		//j += blockDim.y * gridDim.y;
	//}
}

__global__ void calThreadPerColumn(int* output, int numThreads, int k)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int localStorageRegister;


	while (i < GRAPH_SIZE) {

		localStorageRegister = D(i, k);

		for (int j = 0; j < GRAPH_SIZE; j++)
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
__global__ void calcWithAtomic(int* output, int workPerThread)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * workPerThread;
	int j = (blockIdx.y * blockDim.y + threadIdx.y) * workPerThread;
	int k = 0;
	int numBlocks = gridDim.x * gridDim.y;
	//printf("aqui\n");

	while (k < GRAPH_SIZE) {
		//__syncthreads();
		//if (threadIdx.x == 0 && threadIdx.y == 0)
			//printf("Before k = %d\n",k);
		for (int x = i; x < i + workPerThread; x++)
		{
			for (int y = j; y < j + workPerThread; y++)
			{
				//if (x < GRAPH_SIZE && y < GRAPH_SIZE) {
					if (D(x, k) + D(k, y) < D(x, y)) {
						D(x, y) = D(x, k) + D(k, y);
					}
				//}
			}
			//if (threadIdx.x == 0 && threadIdx.y == 0)
			//printf("AFTER: %d\n", x);
		}

		k++;
		/*if (threadIdx.x == 0 && threadIdx.y == 0)
		printf("After\n");
		*/
		//bloco perde-se
		//__syncthreads();
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			atomicAdd(&barrier,1);
			
			//printf("barrier: %d k = %d, blockIdx=%d blockIdy=%d, numblocks: %d\n", barrier,k, blockIdx.x, blockIdx.y, numBlocks);
			while ((atomicCAS(&barrier, numBlocks, numBlocks) % numBlocks) != 0){//antes tinha apenas barrier % numBlocks != 0
			//while((barrier % numBlocks) != 0){
				//printf("barrier %d blockIdx=%d blockIdy=%d\n", barrier, blockIdx.x, blockIdx.y);
				//printf(".");
			}
			//printf("depois while barrier %d k = %d, blockIdx=%d blockIdy=%d, numblocks: %d\n", barrier,k, blockIdx.x, blockIdx.y, numBlocks);
			//barrier = 0;
		}
		__syncthreads();

	}
}

__global__ void calcWithoutAtomic1D(int* output, int k)
{
	int totalID = blockIdx.x * blockDim.x * WORK_SIZE + threadIdx.x * WORK_SIZE;
	int i = totalID / GRAPH_SIZE;
	int j = totalID % GRAPH_SIZE;

	int counter = 0;
	while (counter < WORK_SIZE)
	{
		if (D(i, k) + D(k, j) < D(i, j)) {
			D(i, j) = D(i, k) + D(k, j);
		}
		if (i += ((i + 1) < GRAPH_SIZE)) {
			j++;
		}else {
			j = 0;
		}
		counter++;
	}
}

__global__ void calcWithAtomic1D(int* output)
{
	int totalID = blockIdx.x * blockDim.x * WORK_SIZE + threadIdx.x * WORK_SIZE;
	int i;
	int j;
	int counter;
	int k = 0;
	int numBlocks = gridDim.x;

	while(k < GRAPH_SIZE){
		i = totalID / GRAPH_SIZE;
		j = totalID % GRAPH_SIZE;
		counter = 0;
		while (counter < WORK_SIZE)
		{
			if (D(i, k) + D(k, j) < D(i, j)) {
				D(i, j) = D(i, k) + D(k, j);
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

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			atomicAdd(&barrier,1);
			while ((atomicCAS(&barrier, numBlocks, numBlocks) % numBlocks) != 0);
		}
		__syncthreads();
	}
}

__global__ void calcWithAtomic1DShared(int* output)
{
	int totalID = blockIdx.x * blockDim.x * WORK_SIZE + threadIdx.x * WORK_SIZE;
	int i;
	int j;
	int counter;
	int k = 0;
	int numBlocks = gridDim.x;
	int Dik;
	int Dkj;

	while(k < GRAPH_SIZE){
		i = totalID / GRAPH_SIZE;
		j = totalID % GRAPH_SIZE;
		Dik = D(i,k);
		Dkj = D(k,j);
		counter = 0;
		while (counter < WORK_SIZE)
		{
			if (Dik + Dkj < D(i, j)) {
				D(i, j) = Dik + Dkj;
			}
			if (j + 1 < GRAPH_SIZE) {
				j++;
				Dkj = D(k,j);
			}else {
				i += ((i+1)<GRAPH_SIZE);
				j = 0;
				Dik = D(i,k);
				Dkj = D(k,j);
			}
			counter++;
		}
		k++;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			atomicAdd(&barrier,1);
			while ((atomicCAS(&barrier, numBlocks, numBlocks) % numBlocks) != 0);
			barrier = 0;
		}
		__syncthreads();
	}
}

__global__ void calcWithoutAtomic(int* output, int k, int workPerThread)
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
			if (x < GRAPH_SIZE && y < GRAPH_SIZE) {
				if (D(x, k) + D(k, y) < D(x, y)) {
					D(x, y) = D(x, k) + D(k, y);
				}
			}
		}
	}
}

/*
__global__ void calcSharedWithAtomic(int* output, )
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * WORK_SIZE;
	int j = (blockIdx.y * blockDim.y + threadIdx.y) * WORK_SIZE;
	int k = 0;
	int numBlocks = gridDim.x * gridDim.y;
	
	__shared__ int valuesX[NTHREADS][NTHREADS];
	__shared__ int valuesY[NTHREADS][NTHREADS][WORK_SIZE];

	while (k < GRAPH_SIZE) {
		for (int x = i; x < i + WORK_SIZE; x++)
		{
			if (x < GRAPH_SIZE) {
				valuesX[threadIdx.x][threadIdx.y] = D(x, k);
			}
			for (int y = j; y < j + WORK_SIZE; y++)
			{
				if (x < GRAPH_SIZE && y < GRAPH_SIZE) {
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


__global__ void calcSharedWithoutAtomic(int* output, , int k, int workPerThread)
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
		
		if (x < GRAPH_SIZE) {
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
			if (x < GRAPH_SIZE && y < GRAPH_SIZE) {
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
__global__ void sharedCalcWithoutAtomic(int* output, , int k, const int workPerThread)
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
			if (x < GRAPH_SIZE && y < GRAPH_SIZE) {
				if (D(x, k) + D(k, y) < D(x, y)) {
					D(x, y) = D(x, k) + D(k, y);
				}
			}
		}
	}
}
*/


//sequencial GPU
__global__ void calculateSequencialGPU(int* output)
{
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

void floyd_warshall_gpu(const int* graph, int* output) {

	int* dev_a;
	cudaMalloc(&dev_a, sizeof(int) * GRAPH_SIZE * GRAPH_SIZE);
	cudaMemcpy(dev_a, graph, sizeof(int) * GRAPH_SIZE * GRAPH_SIZE, cudaMemcpyHostToDevice);

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
	int workPerThread = ((GRAPH_SIZE) / maxThreadsPerAxis) + 1;

	fprintf(stderr, "work %d\nthreads %d\n", workPerThread, NThreads);
	*/

	//dim3 threads(NThreads, NThreads);
	//dim3 blocks(maxBlocksPerAxis, maxBlocksPerAxis);

	//printf("blockPerAxis = %d\n", maxBlocksPerAxis);

	//int t = 64;
	//int b = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / t);
	
	//calculateSequencialGPU << <1, 1 >> > (dev_a, GRAPH_SIZE);

	
	int blocks;
	int threads;
	cudaOccupancyMaxPotentialBlockSize (&blocks, &threads, calcWithoutAtomic1D, 0, GRAPH_SIZE*GRAPH_SIZE);
	//blocks = sqrt(blocks);
	//threads = sqrt(threads);
	int workPerThread = ((GRAPH_SIZE*GRAPH_SIZE) / (threads*blocks));// + 1;
	printf("workPerThread= %d, blocks= %d threadsPerBlocks = %d\n", workPerThread, blocks, threads);

	
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
	for (int k = 0; k < GRAPH_SIZE; k++) {
		//calcSharedWithoutAtomic <<<blocks, threads>>> (dev_a, k);
		//calcWithoutAtomic <<<dim3(blocks,blocks), dim3(threads,threads)>>> (dev_a, k, WORK_SIZE);
		calcWithoutAtomic1D <<<BLOCKS, NTHREADS>>> (dev_a, k);
		//calcOnePosPerThread <<<dim3(GRAPH_SIZE/NTHREADS,GRAPH_SIZE/NTHREADS), dim3(NTHREADS,NTHREADS)>>>(dev_a, k);
		//calThreadPerColumn <<<b, t >>> (dev_a, t * b, k);
	}
	*/
	
	
	
	
	//fprintf(stderr, "blocks: %d\nthreads: %d\n", blocks, threads);
	
	//calcWithAtomic <<<dim3(blocks,blocks), dim3(threads,threads)>>> (dev_a, workPerThread);
	//calcWithAtomic1D<<<blocks, threads>>> (dev_a, WORK_SIZE);
	calcWithAtomic1DShared<<<blocks, threads>>> (dev_a);
	//calcSharedWithAtomic <<<blocks, threads >>> (dev_a);

	cudaError_t err = cudaMemcpy(output, dev_a, sizeof(int) * GRAPH_SIZE * GRAPH_SIZE, cudaMemcpyDeviceToHost);
	gpuErrchk(err);
	cudaFree(dev_a);
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
	double interval;
*/ 

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
	//floyd_warshall_cpu(graph, output_cpu);
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

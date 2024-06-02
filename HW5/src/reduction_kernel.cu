#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "reduction.h"

int get_max_threads()
{
    int dev_num;
    int max_threads;
    cudaError_t status;

    // Grab the device number of the default CUDA device.
    status = cudaGetDevice(&dev_num);
    assert(status == cudaSuccess);

    // Query the max possible number of threads per block
    status = cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, dev_num);
    assert(status == cudaSuccess);

    return max_threads;
}

void allocateDeviceMemory(void **M, int size)
{
    cudaError_t err = cudaMalloc(M, size);
    assert(err == cudaSuccess);
}

void deallocateDeviceMemory(void *M)
{
    cudaError_t err = cudaFree(M);
    assert(err == cudaSuccess);
}

void cudaMemcpyToDevice(void *dst, void *src, int size)
{
    cudaError_t err = cudaMemcpy((void *)dst, (void *)src, size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
}

void cudaMemcpyToHost(void *dst, void *src, int size)
{
    cudaError_t err = cudaMemcpy((void *)dst, (void *)src, size, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
}

void reduce_ref(const int *const g_idata, int *const g_odata, const int n)
{
    for (int i = 0; i < n; i++)
        g_odata[0] += g_idata[i];
}

__global__ void reduce1(int *g_idata, int *g_odata, unsigned int n);
__global__ void reduce2(int *g_idata, int *g_odata, unsigned int n);
__global__ void reduce3(int *g_idata, int *g_odata, unsigned int n);
__global__ void reduce4(int *g_idata, int *g_odata, unsigned int n);
__global__ void reduce5(int *g_idata, int *g_odata, unsigned int n);
template <const int blockSize> __global__ void reduce6(int *g_idata, int *g_odata, unsigned int n);
template <const int blockSize> __global__ void reduce7(int *g_idata, int *g_odata, unsigned int n);

void reduce_optimize(const int *const g_idata, int *const g_odata, const int *const d_idata, int *const d_odata,
                     const int n)
{
    // TODO: Implement your CUDA code
    // Reduction result must be stored in d_odata[0]
    // You should run the best kernel in here but you must remain other kernels as evidence.

    const int maxThreads = get_max_threads();

    int threads = (n < maxThreads) ? n : maxThreads;
    int blocks = (n + threads - 1) / threads;
    int remain = blocks / threads;

    reduce1<<<blocks, threads, threads * sizeof(int)>>>((int *)d_idata, d_odata, n);
    reduce1<<<remain, threads, threads * sizeof(int)>>>((int *)d_odata, d_odata, blocks);
    reduce1<<<1, remain, remain * sizeof(int)>>>((int *)d_odata, d_odata, remain);
}

// Reduction #1 : Interleaved Addressing with divergent branching
__global__ void reduce1(int *g_idata, int *g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

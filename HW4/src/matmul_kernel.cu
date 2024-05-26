#include "matmul.h"
#include <assert.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
using namespace std;

#define BLOCK_SIZE 16

__global__ void matmul_kernel(const int *A, const int *B, int *C, int N)
{
    __shared__ int shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int sum = 0;

    for (int i = 0; i < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
    {
        if (i * BLOCK_SIZE + threadIdx.x < N && row < N)
        {
            shared_A[threadIdx.y][threadIdx.x] = A[row * N + i * BLOCK_SIZE + threadIdx.x];
        }
        else
        {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (i * BLOCK_SIZE + threadIdx.y < N && col < N)
        {
            shared_B[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * N + col];
        }
        else
        {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            sum += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
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

void matmul_ref(const int *const matrixA, const int *const matrixB, int *const matrixC, const int n)
{
    // You can assume matrixC is initialized with zero
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

void matmul_optimized(const int *const matrixA, const int *const matrixB, int *const matrixC, const int *d_A,
                      const int *d_B, int *const d_C, const int n)
{
    // TODO: Implement your CUDA code
    int size = n * n * sizeof(int);

    cudaMemcpy((void *)d_A, (void *)matrixA, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_B, (void *)matrixB, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy((void *)matrixC, (void *)d_C, size, cudaMemcpyDeviceToHost);
}

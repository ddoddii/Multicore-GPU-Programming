#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define NUM_BLOCK (128 * 1024)
#define ARRAY_SIZE (1024 * NUM_BLOCK)
#define NUM_STREAMS 4

#define WORK_LOAD 256

// Kenel code
__global__ void myKernel(int *_in, int *_out)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    int temp = 0;
    int in = _in[tID];
    for (int i = 0; i < WORK_LOAD; i++)
    {
        temp = (temp + in * 5) % 10;
    }
    _out[tID] = temp;
}

void main(void)
{
    int *in = NULL, *out = NULL, *dIN = NULL, *dOut = NULL;

    cudaMallocHost(&in, sizeof(int) * ARRAY_SIZE);
    memset(in, 0, sizeof(int) * ARRAY_SIZE);

    cudaMallocHost(&out, sizeof(int) * ARRAY_SIZE);
    memset(out, 0, sizeof(int) * ARRAY_SIZE);

    cudaMalloc(&dIn, sizeof(int) * ARRAY_SIZE);
    cudaMalloc(&dOut, sizeof(int) * ARRAY_SIZE);

    for (int i = 0; i < ARRAY_SIZE; i++)
        in[i] = rand() % 10;

    // Single stream version
    cudaMemcpy(dIn, in, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
    myKernel<<<NUM_BLOCK, 1024>>>(dIn, dOut);
    cudaMemcpy(out, dOut, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

    // Multi-stream version
    cudaStream_t stream[NUM_STREAMS]; // stream 변수 생성
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamCreate(&stream[i]); // Non-NULL stream 생성

    int chunkSize = ARRAY_SIZE / NUM_STREAMS;

    // Copy data to Device using stream
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        int offset = chunkSize * i;
        cudaMemcpyAsync(out + offset, dOut + offset, sizeof(int) * chunckSize, cudaMemcpyHostToDevice, stream[i]);
    }

    // Deliver kenel instructions to each stream
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        int offset = chunkSize * i;
        myKernel<<<NUM_BLOCK / NUM_STREAMS, 1024, 0, stream[i]>>>(dIn + offset, dOut + offset);
    }

    // Copy result data to Host using stream
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        int offset = chunkSize * i;
        cudaMemcpyAsync(out + offset, dOut + offset, sizeof(int) * chunckSize, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamDestory(stream[i]);
    }

    cudaFree(dIn);
    cudaFree(dOut);
    cudaFree(in);
    cudaFree(out);
}
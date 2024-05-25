#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Size of vector
#define NUM_DATA 1024

// Simple vector sum kernel
__global__ void vectorAdd(int *_a, int *_b, int *_c)
{
    int tID = threadIdx.x;
    _c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
    int *a, *b, *c, *hc; // vectors on host(CPU)
    int *da, *db, *dc;   // vectors on device(GPU)

    int memSize = sizeof(int) * NUM_DATA;
    printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

    // Memory allocation on host side
    a = new int[NUM_DATA];
    memset(a, 0, memSize);
    b = new int[NUM_DATA];
    memset(b, 0, memSize);
    c = new int[NUM_DATA];
    memset(c, 0, memSize);
    hc = new int[NUM_DATA];
    memset(hc, 0, memSize);

    // Data generation
    for (int i = 0; i < NUM_DATA; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // Measure time for vector sum on host
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_DATA; i++)
    {
        hc[i] = a[i] + b[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
    printf("CPU time: %f seconds\n", duration_cpu.count());

    // Memory allocation on device-side
    cudaMalloc(&da, memSize);
    cudaMemset(da, 0, memSize);
    cudaMalloc(&db, memSize);
    cudaMemset(db, 0, memSize);
    cudaMalloc(&dc, memSize);
    cudaMemset(dc, 0, memSize);

    // Data copy : Host -> Device
    cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);

    // Measure time for kernel execution
    auto start_gpu = std::chrono::high_resolution_clock::now();
    // Kernel call
    vectorAdd<<<1, NUM_DATA>>>(da, db, dc);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gpu = end_gpu - start_gpu;
    printf("GPU time: %f seconds\n", duration_gpu.count());

    // Copy results : Device -> Host
    cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);

    // Release Device Memory
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    // Check results
    bool result = true;
    for (int i = 0; i < NUM_DATA; i++)
    {
        if (hc[i] != c[i])
        {
            printf("[%d] The result is not matched! (%d,%d)\n", i, hc[i], c[i]);
            result = false;
            break;
        }
    }
    if (result)
        printf("GPU works well !");

    // Release host memory
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] hc;

    return 0;
}
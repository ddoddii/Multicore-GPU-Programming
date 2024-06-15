#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

/*
INPUT PARAMETERS
N : pointer to input array
M : pointer to input mask
P : pointer to output array
Mask_Width : size of the mask
Width : size of the input & output arrays
*/

#define MAX_MASK_WIDTH 10
#define MASK_WIDTH 5

// declare constant variable (Mask array in constant memory)
__constant__ float M[MAX_MASK_WIDTH];

// CUDA kernel function
__global__ void convolution_1D_basic_kernel(float *N, float *P, int Mask_Width, int Width)
{
    // output kernel index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue = 0;
    // Assume MASK_WIDTH is 2*n + 1
    int N_start_point = i - (MASK_WIDTH / 2);
    for (int j = 0; j < MASK_WIDTH; j++)
    {
        // Check boundary condition, Set 0 as default value for ghost cells
        if (N_start_point + j >= 0 && N_start_point + j < Width)
        {
            Pvalue += N[N_start_point + j] * M[j];
        }
    }
    P[i] = Pvalue;
}

// Utility function to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Define array sizes
    const int width = 10;
    const int maskWidth = MASK_WIDTH;

    // Host input arrays
    float h_N[width] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float h_M[maskWidth] = {0.2, 0.4, 0.6, 0.8, 1.0};
    float h_P[width] = {0};

    // Device array
    float *d_N, *d_P;

    // Allocate device memory
    checkCudaError(cudaMalloc((void **)&d_N, width * sizeof(float)), "Allocating d_N");
    checkCudaError(cudaMalloc((void **)&d_P, width * sizeof(float)), "Allocating d_P");

    // Copy host input array to device
    checkCudaError(cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice), "Copying h_N to d_N");

    // Copy mask to constant memory on the device
    checkCudaError(cudaMemcpyToSymbol(M, h_M, maskWidth * sizeof(float)), "Copying h_M to constant memory");

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (width + blockSize - 1) / blockSize;

    // Launch the kernel
    convolution_1D_basic_kernel<<<gridSize, blockSize>>>(d_N, d_P, maskWidth, width);

    // Check for any errors during kernel launch
    checkCudaError(cudaGetLastError(), "Launching kernel");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost), "Copying d_P to h_P");

    // Print the result
    std::cout << "Resultant Array: ";
    for (int i = 0; i < width; i++)
    {
        std::cout << h_P[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    checkCudaError(cudaFree(d_N), "Freeing d_N");
    checkCudaError(cudaFree(d_P), "Freeing d_P");

    return 0;
}
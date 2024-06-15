#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define MAX_MASK_WIDTH 10
#define MASK_WIDTH 5

// declare constant variable (Mask array in constant memory)
__constant__ float M[MAX_MASK_WIDTH];

__global__ void convolution_1D_tiled_caching_kernel(float *N, float *P, int Mask_Width, int Width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float N_ds[TILE_SIZE];

    N_ds[threadIdx.x] = N[i];

    __syncthreads();

    int This_tile_start_point = blockIdx.x * blockDim.x;
    int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
    int N_start_point = i - (Mask_Width / 2);
    float Pvalue = 0;

    for (int j = 0; j < Mask_Width; j++)
    {
        int N_index = N_start_point + j;
        if (N_index >= 0 && N_index < Width)
        {
            if ((N_index >= This_tile_start_point) && (N_index < Next_tile_start_point))
            {
                Pvalue += N_ds[threadIdx.x + j - (Mask_Width / 2)] * M[j];
            }
            else
            {
                Pvalue += N[N_index] * M[j];
            }
        }
    }
    P[i] = Pvalue;
}
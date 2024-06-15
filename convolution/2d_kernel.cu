#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define MAX_MASK_WIDTH 10
#define MASK_WIDTH 5

__global__ void convolution_2D_tiled_kernel(float *P, float *N, int height, int width, int pitch, int channels,
                                            int Mask_Width, const float __restrict__ *M)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.x * O_TILE_WIDTH + ty;
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;

    int row_i = row_o - (Mask_Width / 2);
    int col_i = col_o - (Mask_Width / 2);

    __shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1][TILE_SIZE + MAX_MASK_HEIGHT - 1];

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_o < width))
    {
        N_ds[ty][tx] = data[row_i * pitch + col_i];
    }
    else
    {
        N_ds[ty][tx] = 0.0f;
    }

    float output = 0.0f;
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
    {
        for (int i = 0; i < MASK_WIDTH; i++)
        {
            for (int j = 0; j < MASK_WIDTH; j++)
            {
                output += M[i][j] * N_ds[i + ty][j + tx];
            }
        }
        if (row_o < height && col_o < width)
        {
            data[row_o * width + col_o] = output;
        }
    }
}

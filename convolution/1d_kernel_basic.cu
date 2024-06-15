#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/*
INPUT PARAMETERS
N : pointer to input array
M : pointer to input mask
P : pointer to output array
Mask_Width : size of the mask
Width : size of the input & output arrays
*/

__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width)
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
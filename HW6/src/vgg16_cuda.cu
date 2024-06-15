#include "vgg16_cuda.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#define BLOCK_SIZE 32
#define TILE_WIDTH 16
#define KERNEL_RADIUS 1

const int pool_size = 2;
const int pool_stride = 2;

// Kernel to normalize the input image from uint8 to float
__global__ void normalize_kernel(const uint8_t *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float max_int = 255.0f;
        float mean = 0.5f;
        float var = 0.5f;
        output[idx] = (input[idx] / max_int - mean) / var;
    }
}

// Kernel to pad input feature maps
__global__ void pad_kernel(const float *input, float *output, int batch, int channel, int height, int width,
                           int pad_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_height = height + 2 * pad_size;
    int output_width = width + 2 * pad_size;
    int size = batch * channel * output_height * output_width;

    if (idx < size)
    {
        int c = (idx / (output_height * output_width)) % channel;
        int h = (idx / output_width) % output_height;
        int w = idx % output_width;
        int b = idx / (channel * output_height * output_width);

        int input_h = h - pad_size;
        int input_w = w - pad_size;

        if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width)
        {
            output[idx] = input[b * channel * height * width + c * height * width + input_h * width + input_w];
        }
        else
        {
            output[idx] = 0.0f; // Zero padding
        }
    }
}

// Kernel to perform convolution
__global__ void conv2d_kernel(const float *input, float *output, const float *weight, const float *bias, int batch,
                              int in_channel, int out_channel, int height, int width, int kernel_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;

    if (idx < batch * out_channel * out_height * out_width)
    {
        int w = idx % out_width;
        int h = (idx / out_width) % out_height;
        int oc = (idx / (out_width * out_height)) % out_channel;
        int b = idx / (out_channel * out_height * out_width);

        float sum = bias[oc];
        for (int ic = 0; ic < in_channel; ic++)
        {
            for (int kh = 0; kh < kernel_size; kh++)
            {
                for (int kw = 0; kw < kernel_size; kw++)
                {
                    int input_h = h + kh;
                    int input_w = w + kw;
                    int input_idx = b * in_channel * height * width + ic * height * width + input_h * width + input_w;
                    int weight_idx = oc * in_channel * kernel_size * kernel_size + ic * kernel_size * kernel_size +
                                     kh * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        output[idx] = sum;
    }
}

// Kernel to apply ReLU activation
__global__ void relu_kernel(float *input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        input[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Kernel to perform max pooling
__global__ void max_pooling_kernel(const float *input, float *output, int batch, int channel, int height, int width,
                                   int pool_size, int stride)
{
    int out_height = (height - pool_size) / stride + 1;
    int out_width = (width - pool_size) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * channel * out_height * out_width)
    {
        int w = idx % out_width;
        int h = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channel;
        int b = idx / (channel * out_height * out_width);

        float max_val = -FLT_MAX;
        for (int kh = 0; kh < pool_size; kh++)
        {
            for (int kw = 0; kw < pool_size; kw++)
            {
                int input_h = h * stride + kh;
                int input_w = w * stride + kw;
                int input_idx = b * channel * height * width + c * height * width + input_h * width + input_w;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
        output[idx] = max_val;
    }
}

// Kernel to perform fully connected layer
__global__ void fc_layer_kernel(const float *input, float *output, const float *weight, const float *bias, int batch,
                                int input_size, int output_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * output_size)
    {
        int out_idx = idx % output_size;
        int b = idx / output_size;

        float sum = bias[out_idx];
        for (int in_idx = 0; in_idx < input_size; in_idx++)
        {
            sum += input[b * input_size + in_idx] * weight[out_idx * input_size + in_idx];
        }
        output[idx] = sum;
    }
}

// Helper function to print device array (for debugging) - 1
void print_device_array(const char *label, const float *d_array, int size)
{
    float *h_array = (float *)malloc(size * sizeof(float));
    cudaMemcpy(h_array, d_array, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[DEBUG CUDA] %s:\n", label);
    for (int i = 0; i < size; i++)
    {
        printf("%f ", h_array[i]);
        if ((i + 1) % 10 == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
    free(h_array);
}

// Helper function to print device array (for debugging) - 2
void print_device_array(const char *message, float *d_array, int size, int start_idx = 0, int end_idx = 100)
{
    std::vector<float> h_array(size);
    cudaMemcpy(h_array.data(), d_array, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[DEBUG] %s:\n", message);
    for (int i = start_idx; i < end_idx && i < size; ++i)
    {
        printf("%f ", h_array[i]);
        if ((i + 1) % 10 == 0)
            printf("\n");
    }
    printf("\n");
}

void vgg16_cuda::predict(int batch)
{
    dim3 threadsPerBlock(256);
    // dim3 numBlocks;

    // Normalize
    dim3 numBlocks((batch * input_channel * input_size * input_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    normalize_kernel<<<numBlocks, threadsPerBlock>>>(d_image, d_input, batch * input_channel * input_size * input_size);
    cudaDeviceSynchronize();

    //////////BLOCK 1/////////////////////////////////
    // Pad
    int padded_input_size =
        batch * input_channel * (input_size + 2 * conv1_1_padding_size) * (input_size + 2 * conv1_1_padding_size);
    numBlocks = dim3((padded_input_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_input_padded, batch, input_channel, input_size, input_size,
                                               conv1_1_padding_size);
    cudaDeviceSynchronize();

    // Convolution 1-1
    int conv1_1_output_size = batch * C1_1_channel * C1_1_size * C1_1_size;
    numBlocks = dim3((conv1_1_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_input_padded, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias,
                                                  batch, conv1_1_in_channel, conv1_1_out_channel,
                                                  input_size + 2 * conv1_1_padding_size,
                                                  input_size + 2 * conv1_1_padding_size, conv1_1_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C1_1_feature_map, conv1_1_output_size);
    cudaDeviceSynchronize();

    // Pad
    int padded_C1_1_size =
        batch * C1_1_channel * (C1_1_size + 2 * conv1_2_padding_size) * (C1_1_size + 2 * conv1_2_padding_size);
    numBlocks = dim3((padded_C1_1_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_C1_1_feature_map, d_C1_1_feature_map_padded, batch, C1_1_channel,
                                               C1_1_size, C1_1_size, conv1_2_padding_size);
    cudaDeviceSynchronize();

    // Convolution 1-2
    int conv1_2_output_size = batch * C1_2_channel * C1_2_size * C1_2_size;
    numBlocks = dim3((conv1_2_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_C1_1_feature_map_padded, d_C1_2_feature_map, d_conv1_2_weight,
                                                  d_conv1_2_bias, batch, conv1_2_in_channel, conv1_2_out_channel,
                                                  C1_1_size + 2 * conv1_2_padding_size,
                                                  C1_1_size + 2 * conv1_2_padding_size, conv1_2_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C1_2_feature_map, conv1_2_output_size);
    cudaDeviceSynchronize();

    // Pooling
    int pool1_output_size = batch * S1_channel * S1_size * S1_size;
    numBlocks = dim3((pool1_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    max_pooling_kernel<<<numBlocks, threadsPerBlock>>>(d_C1_2_feature_map, d_S1_feature_map, batch, S1_channel,
                                                       C1_2_size, C1_2_size, pool_size, pool_stride);
    cudaDeviceSynchronize();

    /*BLOCK 1 DEBUG*/
    print_device_array("After Normalization", d_input, batch * input_channel * input_size * input_size);
    print_device_array("After Padding (Block 1)", d_input_padded, padded_input_size);
    print_device_array("After Convolution 1-1", d_C1_1_feature_map, conv1_1_output_size);
    print_device_array("After ReLU 1-1", d_C1_1_feature_map, conv1_1_output_size);
    print_device_array("After Padding 1-2", d_C1_1_feature_map_padded, padded_C1_1_size);
    print_device_array("After Convolution 1-2", d_C1_2_feature_map, conv1_2_output_size);
    print_device_array("After ReLU 1-2", d_C1_2_feature_map, conv1_2_output_size);
    print_device_array("After Pooling (Block 1)", d_S1_feature_map, pool1_output_size);

    //////////BLOCK 2/////////////////////////////////
    // Pad
    int padded_S1_size =
        batch * S1_channel * (S1_size + 2 * conv2_1_padding_size) * (S1_size + 2 * conv2_1_padding_size);
    numBlocks = dim3((padded_S1_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_S1_feature_map, d_S1_feature_map_padded, batch, S1_channel, S1_size,
                                               S1_size, conv2_1_padding_size);
    cudaDeviceSynchronize();

    // Convolution 2-1
    int conv2_1_output_size = batch * C2_1_channel * C2_1_size * C2_1_size;
    numBlocks = dim3((conv2_1_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_S1_feature_map_padded, d_C2_1_feature_map, d_conv2_1_weight,
                                                  d_conv2_1_bias, batch, conv2_1_in_channel, conv2_1_out_channel,
                                                  S1_size + 2 * conv2_1_padding_size,
                                                  S1_size + 2 * conv2_1_padding_size, conv2_1_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C2_1_feature_map, conv2_1_output_size);
    cudaDeviceSynchronize();

    // Pad
    int padded_C2_1_size =
        batch * C2_1_channel * (C2_1_size + 2 * conv2_2_padding_size) * (C2_1_size + 2 * conv2_2_padding_size);
    numBlocks = dim3((padded_C2_1_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_C2_1_feature_map, d_C2_1_feature_map_padded, batch, C2_1_channel,
                                               C2_1_size, C2_1_size, conv2_2_padding_size);
    cudaDeviceSynchronize();

    // Convolution 2-2
    int conv2_2_output_size = batch * C2_2_channel * C2_2_size * C2_2_size;
    numBlocks = dim3((conv2_2_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_C2_1_feature_map_padded, d_C2_2_feature_map, d_conv2_2_weight,
                                                  d_conv2_2_bias, batch, conv2_2_in_channel, conv2_2_out_channel,
                                                  C2_1_size + 2 * conv2_2_padding_size,
                                                  C2_1_size + 2 * conv2_2_padding_size, conv2_2_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C2_2_feature_map, conv2_2_output_size);
    cudaDeviceSynchronize();

    // Pooling
    int pool2_output_size = batch * S2_channel * S2_size * S2_size;
    numBlocks = dim3((pool2_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    max_pooling_kernel<<<numBlocks, threadsPerBlock>>>(d_C2_2_feature_map, d_S2_feature_map, batch, S2_channel,
                                                       C2_2_size, C2_2_size, pool_size, pool_stride);
    cudaDeviceSynchronize();

    /*DEBUG*/
    print_device_array("After Padding (Block 2)", d_S1_feature_map_padded, padded_S1_size);
    print_device_array("After Convolution 2-1", d_C2_1_feature_map, conv2_1_output_size);
    print_device_array("After ReLU 2-1", d_C2_1_feature_map, conv2_1_output_size);
    print_device_array("After Padding 2-2", d_C2_1_feature_map_padded, padded_C2_1_size);
    print_device_array("After Convolution 2-2", d_C2_2_feature_map, conv2_2_output_size);
    print_device_array("After ReLU 2-2", d_C2_2_feature_map, conv2_2_output_size);
    print_device_array("After Pooling (Block 2)", d_S2_feature_map, pool2_output_size);

    //////////BLOCK 3/////////////////////////////////
    // Pad
    int padded_S2_size =
        batch * S2_channel * (S2_size + 2 * conv3_1_padding_size) * (S2_size + 2 * conv3_1_padding_size);
    numBlocks = dim3((padded_S2_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_S2_feature_map, d_S2_feature_map_padded, batch, S2_channel, S2_size,
                                               S2_size, conv3_1_padding_size);
    cudaDeviceSynchronize();

    // Convolution 3-1
    int conv3_1_output_size = batch * C3_1_channel * C3_1_size * C3_1_size;
    numBlocks = dim3((conv3_1_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_S2_feature_map_padded, d_C3_1_feature_map, d_conv3_1_weight,
                                                  d_conv3_1_bias, batch, conv3_1_in_channel, conv3_1_out_channel,
                                                  S2_size + 2 * conv3_1_padding_size,
                                                  S2_size + 2 * conv3_1_padding_size, conv3_1_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C3_1_feature_map, conv3_1_output_size);
    cudaDeviceSynchronize();

    // Pad
    int padded_C3_1_size =
        batch * C3_1_channel * (C3_1_size + 2 * conv3_2_padding_size) * (C3_1_size + 2 * conv3_2_padding_size);
    numBlocks = dim3((padded_C3_1_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_C3_1_feature_map, d_C3_1_feature_map_padded, batch, C3_1_channel,
                                               C3_1_size, C3_1_size, conv3_2_padding_size);
    cudaDeviceSynchronize();

    // Convolution 3-2
    int conv3_2_output_size = batch * C3_2_channel * C3_2_size * C3_2_size;
    numBlocks = dim3((conv3_2_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_C3_1_feature_map_padded, d_C3_2_feature_map, d_conv3_2_weight,
                                                  d_conv3_2_bias, batch, conv3_2_in_channel, conv3_2_out_channel,
                                                  C3_1_size + 2 * conv3_2_padding_size,
                                                  C3_1_size + 2 * conv3_2_padding_size, conv3_2_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C3_2_feature_map, conv3_2_output_size);
    cudaDeviceSynchronize();

    // Pad
    int padded_C3_2_size =
        batch * C3_2_channel * (C3_2_size + 2 * conv3_3_padding_size) * (C3_2_size + 2 * conv3_3_padding_size);
    numBlocks = dim3((padded_C3_2_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_C3_2_feature_map, d_C3_2_feature_map_padded, batch, C3_2_channel,
                                               C3_2_size, C3_2_size, conv3_3_padding_size);
    cudaDeviceSynchronize();

    // Convolution 3-3
    int conv3_3_output_size = batch * C3_3_channel * C3_3_size * C3_3_size;
    numBlocks = dim3((conv3_3_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_C3_2_feature_map_padded, d_C3_3_feature_map, d_conv3_3_weight,
                                                  d_conv3_3_bias, batch, conv3_3_in_channel, conv3_3_out_channel,
                                                  C3_2_size + 2 * conv3_3_padding_size,
                                                  C3_2_size + 2 * conv3_3_padding_size, conv3_3_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C3_3_feature_map, conv3_3_output_size);
    cudaDeviceSynchronize();

    // Pooling
    int pool3_output_size = batch * S3_channel * S3_size * S3_size;
    numBlocks = dim3((pool3_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    max_pooling_kernel<<<numBlocks, threadsPerBlock>>>(d_C3_3_feature_map, d_S3_feature_map, batch, S3_channel,
                                                       C3_3_size, C3_3_size, pool_size, pool_stride);
    cudaDeviceSynchronize();

    /*DEBUG*/
    print_device_array("After Padding (Block 3)", d_S2_feature_map_padded, padded_S2_size);
    print_device_array("After Convolution 3-1", d_C3_1_feature_map, conv3_1_output_size);
    print_device_array("After ReLU 3-1", d_C3_1_feature_map, conv3_1_output_size);
    print_device_array("After Padding 3-2", d_C3_1_feature_map_padded, padded_C3_1_size);
    print_device_array("After Convolution 3-2", d_C3_2_feature_map, conv3_2_output_size);
    print_device_array("After ReLU 3-2", d_C3_2_feature_map, conv3_2_output_size);
    print_device_array("After Pooling (Block 3)", d_S3_feature_map, pool3_output_size);

    //////////BLOCK 4/////////////////////////////////
    // Pad
    int padded_S3_size =
        batch * S3_channel * (S3_size + 2 * conv4_1_padding_size) * (S3_size + 2 * conv4_1_padding_size);
    numBlocks = dim3((padded_S3_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_S3_feature_map, d_S3_feature_map_padded, batch, S3_channel, S3_size,
                                               S3_size, conv4_1_padding_size);
    cudaDeviceSynchronize();

    // Convolution 4-1
    int conv4_1_output_size = batch * C4_1_channel * C4_1_size * C4_1_size;
    numBlocks = dim3((conv4_1_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_S3_feature_map_padded, d_C4_1_feature_map, d_conv4_1_weight,
                                                  d_conv4_1_bias, batch, conv4_1_in_channel, conv4_1_out_channel,
                                                  S3_size + 2 * conv4_1_padding_size,
                                                  S3_size + 2 * conv4_1_padding_size, conv4_1_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C4_1_feature_map, conv4_1_output_size);
    cudaDeviceSynchronize();

    // Pad
    int padded_C4_1_size =
        batch * C4_1_channel * (C4_1_size + 2 * conv4_2_padding_size) * (C4_1_size + 2 * conv4_2_padding_size);
    numBlocks = dim3((padded_C4_1_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_C4_1_feature_map, d_C4_1_feature_map_padded, batch, C4_1_channel,
                                               C4_1_size, C4_1_size, conv4_2_padding_size);
    cudaDeviceSynchronize();

    // Convolution 4-2
    int conv4_2_output_size = batch * C4_2_channel * C4_2_size * C4_2_size;
    numBlocks = dim3((conv4_2_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_C4_1_feature_map_padded, d_C4_2_feature_map, d_conv4_2_weight,
                                                  d_conv4_2_bias, batch, conv4_2_in_channel, conv4_2_out_channel,
                                                  C4_1_size + 2 * conv4_2_padding_size,
                                                  C4_1_size + 2 * conv4_2_padding_size, conv4_2_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C4_2_feature_map, conv4_2_output_size);
    cudaDeviceSynchronize();

    // Pad
    int padded_C4_2_size =
        batch * C4_2_channel * (C4_2_size + 2 * conv4_3_padding_size) * (C4_2_size + 2 * conv4_3_padding_size);
    numBlocks = dim3((padded_C4_2_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_C4_2_feature_map, d_C4_2_feature_map_padded, batch, C4_2_channel,
                                               C4_2_size, C4_2_size, conv4_3_padding_size);
    cudaDeviceSynchronize();

    // Convolution 4-3
    int conv4_3_output_size = batch * C4_3_channel * C4_3_size * C4_3_size;
    numBlocks = dim3((conv4_3_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_C4_2_feature_map_padded, d_C4_3_feature_map, d_conv4_3_weight,
                                                  d_conv4_3_bias, batch, conv4_3_in_channel, conv4_3_out_channel,
                                                  C4_2_size + 2 * conv4_3_padding_size,
                                                  C4_2_size + 2 * conv4_3_padding_size, conv4_3_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C4_3_feature_map, conv4_3_output_size);
    cudaDeviceSynchronize();

    // Pooling
    int pool4_output_size = batch * S4_channel * S4_size * S4_size;
    numBlocks = dim3((pool4_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    max_pooling_kernel<<<numBlocks, threadsPerBlock>>>(d_C4_3_feature_map, d_S4_feature_map, batch, S4_channel,
                                                       C4_3_size, C4_3_size, pool_size, pool_stride);
    cudaDeviceSynchronize();

    /*DEBUG*/
    print_device_array("After Padding (Block 4)", d_S3_feature_map_padded, padded_S3_size);
    print_device_array("After Convolution 4-1", d_C4_1_feature_map, conv4_1_output_size);
    print_device_array("After ReLU 4-1", d_C4_1_feature_map, conv4_1_output_size);
    print_device_array("After Padding 4-2", d_C4_1_feature_map_padded, padded_C4_1_size);
    print_device_array("After Convolution 4-2", d_C4_2_feature_map, conv4_2_output_size);
    print_device_array("After ReLU 4-2", d_C4_2_feature_map, conv4_2_output_size);
    print_device_array("After Pooling (Block 4)", d_S4_feature_map, pool4_output_size);

    //////////BLOCK 5/////////////////////////////////
    // Pad
    int padded_S4_size =
        batch * S4_channel * (S4_size + 2 * conv5_1_padding_size) * (S4_size + 2 * conv5_1_padding_size);
    numBlocks = dim3((padded_S4_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_S4_feature_map, d_S4_feature_map_padded, batch, S4_channel, S4_size,
                                               S4_size, conv5_1_padding_size);
    cudaDeviceSynchronize();

    // Convolution 5-1
    int conv5_1_output_size = batch * C5_1_channel * C5_1_size * C5_1_size;
    numBlocks = dim3((conv5_1_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_S4_feature_map_padded, d_C5_1_feature_map, d_conv5_1_weight,
                                                  d_conv5_1_bias, batch, conv5_1_in_channel, conv5_1_out_channel,
                                                  S4_size + 2 * conv5_1_padding_size,
                                                  S4_size + 2 * conv5_1_padding_size, conv5_1_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C5_1_feature_map, conv5_1_output_size);
    cudaDeviceSynchronize();

    // Pad
    int padded_C5_1_size =
        batch * C5_1_channel * (C5_1_size + 2 * conv5_2_padding_size) * (C5_1_size + 2 * conv5_2_padding_size);
    numBlocks = dim3((padded_C5_1_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_C5_1_feature_map, d_C5_1_feature_map_padded, batch, C5_1_channel,
                                               C5_1_size, C5_1_size, conv5_2_padding_size);
    cudaDeviceSynchronize();

    // Convolution 5-2
    int conv5_2_output_size = batch * C5_2_channel * C5_2_size * C5_2_size;
    numBlocks = dim3((conv5_2_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_C5_1_feature_map_padded, d_C5_2_feature_map, d_conv5_2_weight,
                                                  d_conv5_2_bias, batch, conv5_2_in_channel, conv5_2_out_channel,
                                                  C5_1_size + 2 * conv5_2_padding_size,
                                                  C5_1_size + 2 * conv5_2_padding_size, conv5_2_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C5_2_feature_map, conv5_2_output_size);
    cudaDeviceSynchronize();

    // Pad
    int padded_C5_2_size =
        batch * C5_2_channel * (C5_2_size + 2 * conv5_3_padding_size) * (C5_2_size + 2 * conv5_3_padding_size);
    numBlocks = dim3((padded_C5_2_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pad_kernel<<<numBlocks, threadsPerBlock>>>(d_C5_2_feature_map, d_C5_2_feature_map_padded, batch, C5_2_channel,
                                               C5_2_size, C5_2_size, conv5_3_padding_size);
    cudaDeviceSynchronize();

    // Convolution 5-3
    int conv5_3_output_size = batch * C5_3_channel * C5_3_size * C5_3_size;
    numBlocks = dim3((conv5_3_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_C5_2_feature_map_padded, d_C5_3_feature_map, d_conv5_3_weight,
                                                  d_conv5_3_bias, batch, conv5_3_in_channel, conv5_3_out_channel,
                                                  C5_2_size + 2 * conv5_3_padding_size,
                                                  C5_2_size + 2 * conv5_3_padding_size, conv5_3_kernel_size);
    cudaDeviceSynchronize();

    // ReLU
    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_C5_3_feature_map, conv5_3_output_size);
    cudaDeviceSynchronize();

    // Pooling
    int pool5_output_size = batch * S5_channel * S5_size * S5_size;
    numBlocks = dim3((pool5_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    max_pooling_kernel<<<numBlocks, threadsPerBlock>>>(d_C5_3_feature_map, d_S5_feature_map, batch, S5_channel,
                                                       C5_3_size, C5_3_size, pool_size, pool_stride);
    cudaDeviceSynchronize();

    /* BLOCK 5 DEBUG*/
    print_device_array("After Padding (Block 5)", d_S4_feature_map_padded, padded_S4_size);
    print_device_array("After Convolution 5-1", d_C5_1_feature_map, conv5_1_output_size);
    print_device_array("After ReLU 5-1", d_C5_1_feature_map, conv5_1_output_size);
    print_device_array("After Padding 5-2", d_C5_1_feature_map_padded, padded_C5_1_size);
    print_device_array("After Convolution 5-2", d_C5_2_feature_map, conv5_2_output_size);
    print_device_array("After ReLU 5-2", d_C5_2_feature_map, conv5_2_output_size);
    print_device_array("After Pooling (Block 5)", d_S5_feature_map, pool5_output_size);

    //////////Fully Connected Layer/////////////////////////////////
    int fc1_output_size = batch * fc1_out_channel;
    numBlocks = dim3((fc1_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    fc_layer_kernel<<<numBlocks, threadsPerBlock>>>(d_S5_feature_map, d_output, d_fc1_weight, d_fc1_bias, batch,
                                                    fc1_in_channel, fc1_out_channel);
    cudaDeviceSynchronize();

    // ReLU
    // relu_kernel<<<numBlocks, threadsPerBlock>>>(d_output, fc1_output_size);
    // cudaDeviceSynchronize();
}

void vgg16_cuda::prepare_device_memory(uint8_t *image)
{
    // Alloc Model Parameters

    //////////BLOCK 1/////////////////////////////////
    cudaMalloc((void **)&d_conv1_1_weight,
               sizeof(float) * conv1_1_in_channel * conv1_1_out_channel * conv1_1_kernel_size * conv1_1_kernel_size);
    cudaMalloc((void **)&d_conv1_1_bias, sizeof(float) * conv1_1_out_channel);
    cudaMalloc((void **)&d_conv1_2_weight,
               sizeof(float) * conv1_2_in_channel * conv1_2_out_channel * conv1_2_kernel_size * conv1_2_kernel_size);
    cudaMalloc((void **)&d_conv1_2_bias, sizeof(float) * conv1_2_out_channel);

    //////////BLOCK 2/////////////////////////////////
    cudaMalloc((void **)&d_conv2_1_weight,
               sizeof(float) * conv2_1_in_channel * conv2_1_out_channel * conv2_1_kernel_size * conv2_1_kernel_size);
    cudaMalloc((void **)&d_conv2_1_bias, sizeof(float) * conv2_1_out_channel);
    cudaMalloc((void **)&d_conv2_2_weight,
               sizeof(float) * conv2_2_in_channel * conv2_2_out_channel * conv2_2_kernel_size * conv2_2_kernel_size);
    cudaMalloc((void **)&d_conv2_2_bias, sizeof(float) * conv2_2_out_channel);

    //////////BLOCK 3/////////////////////////////////
    cudaMalloc((void **)&d_conv3_1_weight,
               sizeof(float) * conv3_1_in_channel * conv3_1_out_channel * conv3_1_kernel_size * conv3_1_kernel_size);
    cudaMalloc((void **)&d_conv3_1_bias, sizeof(float) * conv3_1_out_channel);
    cudaMalloc((void **)&d_conv3_2_weight,
               sizeof(float) * conv3_2_in_channel * conv3_2_out_channel * conv3_2_kernel_size * conv3_2_kernel_size);
    cudaMalloc((void **)&d_conv3_2_bias, sizeof(float) * conv3_2_out_channel);
    cudaMalloc((void **)&d_conv3_3_weight,
               sizeof(float) * conv3_3_in_channel * conv3_3_out_channel * conv3_3_kernel_size * conv3_3_kernel_size);
    cudaMalloc((void **)&d_conv3_3_bias, sizeof(float) * conv3_3_out_channel);

    //////////BLOCK 4/////////////////////////////////
    cudaMalloc((void **)&d_conv4_1_weight,
               sizeof(float) * conv4_1_in_channel * conv4_1_out_channel * conv4_1_kernel_size * conv4_1_kernel_size);
    cudaMalloc((void **)&d_conv4_1_bias, sizeof(float) * conv4_1_out_channel);
    cudaMalloc((void **)&d_conv4_2_weight,
               sizeof(float) * conv4_2_in_channel * conv4_2_out_channel * conv4_2_kernel_size * conv4_2_kernel_size);
    cudaMalloc((void **)&d_conv4_2_bias, sizeof(float) * conv4_2_out_channel);
    cudaMalloc((void **)&d_conv4_3_weight,
               sizeof(float) * conv4_3_in_channel * conv4_3_out_channel * conv4_3_kernel_size * conv4_3_kernel_size);
    cudaMalloc((void **)&d_conv4_3_bias, sizeof(float) * conv4_3_out_channel);

    //////////BLOCK 5/////////////////////////////////
    cudaMalloc((void **)&d_conv5_1_weight,
               sizeof(float) * conv5_1_in_channel * conv5_1_out_channel * conv5_1_kernel_size * conv5_1_kernel_size);
    cudaMalloc((void **)&d_conv5_1_bias, sizeof(float) * conv5_1_out_channel);
    cudaMalloc((void **)&d_conv5_2_weight,
               sizeof(float) * conv5_2_in_channel * conv5_2_out_channel * conv5_2_kernel_size * conv5_2_kernel_size);
    cudaMalloc((void **)&d_conv5_2_bias, sizeof(float) * conv5_2_out_channel);
    cudaMalloc((void **)&d_conv5_3_weight,
               sizeof(float) * conv5_3_in_channel * conv5_3_out_channel * conv5_3_kernel_size * conv5_3_kernel_size);
    cudaMalloc((void **)&d_conv5_3_bias, sizeof(float) * conv5_3_out_channel);

    //////////FC 1////////////////////////////////////
    cudaMalloc((void **)&d_fc1_weight, sizeof(float) * fc1_in_channel * fc1_out_channel);
    cudaMalloc((void **)&d_fc1_bias, sizeof(float) * fc1_out_channel);

    // Alloc Activations
    cudaMalloc((void **)&d_image, sizeof(uint8_t) * batch * input_size * input_size * input_channel);
    cudaMalloc((void **)&d_input, sizeof(float) * batch * input_channel * input_size * input_size);

    //////////BLOCK 1/////////////////////////////////
    cudaMalloc((void **)&d_input_padded, sizeof(float) * batch * input_channel *
                                             (input_size + 2 * conv1_1_padding_size) *
                                             (input_size + 2 * conv1_1_padding_size));
    cudaMalloc((void **)&d_C1_1_feature_map, sizeof(float) * batch * C1_1_channel * C1_1_size * C1_1_size);
    cudaMalloc((void **)&d_C1_1_feature_map_padded, sizeof(float) * batch * C1_1_channel *
                                                        (C1_1_size + 2 * conv1_2_padding_size) *
                                                        (C1_1_size + 2 * conv1_2_padding_size));
    cudaMalloc((void **)&d_C1_2_feature_map, sizeof(float) * batch * C1_2_channel * C1_2_size * C1_2_size);
    cudaMalloc((void **)&d_S1_feature_map, sizeof(float) * batch * S1_channel * S1_size * S1_size);

    //////////BLOCK 2/////////////////////////////////
    cudaMalloc((void **)&d_S1_feature_map_padded, sizeof(float) * batch * S1_channel *
                                                      (S1_size + 2 * conv2_1_padding_size) *
                                                      (S1_size + 2 * conv2_1_padding_size));
    cudaMalloc((void **)&d_C2_1_feature_map, sizeof(float) * batch * C2_1_channel * C2_1_size * C2_1_size);
    cudaMalloc((void **)&d_C2_1_feature_map_padded, sizeof(float) * batch * C2_1_channel *
                                                        (C2_1_size + 2 * conv2_2_padding_size) *
                                                        (C2_1_size + 2 * conv2_2_padding_size));
    cudaMalloc((void **)&d_C2_2_feature_map, sizeof(float) * batch * C2_2_channel * C2_2_size * C2_2_size);
    cudaMalloc((void **)&d_S2_feature_map, sizeof(float) * batch * S2_channel * S2_size * S2_size);

    //////////BLOCK 3/////////////////////////////////
    cudaMalloc((void **)&d_S2_feature_map_padded, sizeof(float) * batch * S2_channel *
                                                      (S2_size + 2 * conv3_1_padding_size) *
                                                      (S2_size + 2 * conv3_1_padding_size));
    cudaMalloc((void **)&d_C3_1_feature_map, sizeof(float) * batch * C3_1_channel * C3_1_size * C3_1_size);
    cudaMalloc((void **)&d_C3_1_feature_map_padded, sizeof(float) * batch * C3_1_channel *
                                                        (C3_1_size + 2 * conv3_2_padding_size) *
                                                        (C3_1_size + 2 * conv3_2_padding_size));
    cudaMalloc((void **)&d_C3_2_feature_map, sizeof(float) * batch * C3_2_channel * C3_2_size * C3_2_size);
    cudaMalloc((void **)&d_C3_2_feature_map_padded, sizeof(float) * batch * C3_2_channel *
                                                        (C3_2_size + 2 * conv3_3_padding_size) *
                                                        (C3_2_size + 2 * conv3_3_padding_size));
    cudaMalloc((void **)&d_C3_3_feature_map, sizeof(float) * batch * C3_3_channel * C3_3_size * C3_3_size);
    cudaMalloc((void **)&d_S3_feature_map, sizeof(float) * batch * S3_channel * S3_size * S3_size);

    //////////BLOCK 4/////////////////////////////////
    cudaMalloc((void **)&d_S3_feature_map_padded, sizeof(float) * batch * S3_channel *
                                                      (S3_size + 2 * conv4_1_padding_size) *
                                                      (S3_size + 2 * conv4_1_padding_size));
    cudaMalloc((void **)&d_C4_1_feature_map, sizeof(float) * batch * C4_1_channel * C4_1_size * C4_1_size);
    cudaMalloc((void **)&d_C4_1_feature_map_padded, sizeof(float) * batch * C4_1_channel *
                                                        (C4_1_size + 2 * conv4_2_padding_size) *
                                                        (C4_1_size + 2 * conv4_2_padding_size));
    cudaMalloc((void **)&d_C4_2_feature_map, sizeof(float) * batch * C4_2_channel * C4_2_size * C4_2_size);
    cudaMalloc((void **)&d_C4_2_feature_map_padded, sizeof(float) * batch * C4_2_channel *
                                                        (C4_2_size + 2 * conv4_3_padding_size) *
                                                        (C4_2_size + 2 * conv4_3_padding_size));
    cudaMalloc((void **)&d_C4_3_feature_map, sizeof(float) * batch * C4_3_channel * C4_3_size * C4_3_size);
    cudaMalloc((void **)&d_S4_feature_map, sizeof(float) * batch * S4_channel * S4_size * S4_size);

    //////////BLOCK 5/////////////////////////////////
    cudaMalloc((void **)&d_S4_feature_map_padded, sizeof(float) * batch * S4_channel *
                                                      (S4_size + 2 * conv5_1_padding_size) *
                                                      (S4_size + 2 * conv5_1_padding_size));
    cudaMalloc((void **)&d_C5_1_feature_map, sizeof(float) * batch * C5_1_channel * C5_1_size * C5_1_size);
    cudaMalloc((void **)&d_C5_1_feature_map_padded, sizeof(float) * batch * C5_1_channel *
                                                        (C5_1_size + 2 * conv5_2_padding_size) *
                                                        (C5_1_size + 2 * conv5_2_padding_size));
    cudaMalloc((void **)&d_C5_2_feature_map, sizeof(float) * batch * C5_2_channel * C5_2_size * C5_2_size);
    cudaMalloc((void **)&d_C5_2_feature_map_padded, sizeof(float) * batch * C5_2_channel *
                                                        (C5_2_size + 2 * conv5_3_padding_size) *
                                                        (C5_2_size + 2 * conv5_3_padding_size));
    cudaMalloc((void **)&d_C5_3_feature_map, sizeof(float) * batch * C5_3_channel * C5_3_size * C5_3_size);
    cudaMalloc((void **)&d_S5_feature_map, sizeof(float) * batch * S5_channel * S5_size * S5_size);

    cudaMalloc((void **)&d_output, sizeof(float) * batch * output_size);

    // Copy Parameters
    //////////BLOCK 1/////////////////////////////////
    cudaMemcpy(d_conv1_1_weight, conv1_1_weight,
               sizeof(float) * conv1_1_in_channel * conv1_1_out_channel * conv1_1_kernel_size * conv1_1_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_1_bias, conv1_1_bias, sizeof(float) * conv1_1_out_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_2_weight, conv1_2_weight,
               sizeof(float) * conv1_2_in_channel * conv1_2_out_channel * conv1_2_kernel_size * conv1_2_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_2_bias, conv1_2_bias, sizeof(float) * conv1_2_out_channel, cudaMemcpyHostToDevice);

    //////////BLOCK 2/////////////////////////////////
    cudaMemcpy(d_conv2_1_weight, conv2_1_weight,
               sizeof(float) * conv2_1_in_channel * conv2_1_out_channel * conv2_1_kernel_size * conv2_1_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_1_bias, conv2_1_bias, sizeof(float) * conv2_1_out_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_2_weight, conv2_2_weight,
               sizeof(float) * conv2_2_in_channel * conv2_2_out_channel * conv2_2_kernel_size * conv2_2_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_2_bias, conv2_2_bias, sizeof(float) * conv2_2_out_channel, cudaMemcpyHostToDevice);

    //////////BLOCK 3/////////////////////////////////
    cudaMemcpy(d_conv3_1_weight, conv3_1_weight,
               sizeof(float) * conv3_1_in_channel * conv3_1_out_channel * conv3_1_kernel_size * conv3_1_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_1_bias, conv3_1_bias, sizeof(float) * conv3_1_out_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_2_weight, conv3_2_weight,
               sizeof(float) * conv3_2_in_channel * conv3_2_out_channel * conv3_2_kernel_size * conv3_2_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_2_bias, conv3_2_bias, sizeof(float) * conv3_2_out_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_3_weight, conv3_3_weight,
               sizeof(float) * conv3_3_in_channel * conv3_3_out_channel * conv3_3_kernel_size * conv3_3_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_3_bias, conv3_3_bias, sizeof(float) * conv3_3_out_channel, cudaMemcpyHostToDevice);

    //////////BLOCK 4/////////////////////////////////
    cudaMemcpy(d_conv4_1_weight, conv4_1_weight,
               sizeof(float) * conv4_1_in_channel * conv4_1_out_channel * conv4_1_kernel_size * conv4_1_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_1_bias, conv4_1_bias, sizeof(float) * conv4_1_out_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_2_weight, conv4_2_weight,
               sizeof(float) * conv4_2_in_channel * conv4_2_out_channel * conv4_2_kernel_size * conv4_2_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_2_bias, conv4_2_bias, sizeof(float) * conv4_2_out_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_3_weight, conv4_3_weight,
               sizeof(float) * conv4_3_in_channel * conv4_3_out_channel * conv4_3_kernel_size * conv4_3_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_3_bias, conv4_3_bias, sizeof(float) * conv4_3_out_channel, cudaMemcpyHostToDevice);

    //////////BLOCK 5/////////////////////////////////
    cudaMemcpy(d_conv5_1_weight, conv5_1_weight,
               sizeof(float) * conv5_1_in_channel * conv5_1_out_channel * conv5_1_kernel_size * conv5_1_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_1_bias, conv5_1_bias, sizeof(float) * conv5_1_out_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_2_weight, conv5_2_weight,
               sizeof(float) * conv5_2_in_channel * conv5_2_out_channel * conv5_2_kernel_size * conv5_2_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_2_bias, conv5_2_bias, sizeof(float) * conv5_2_out_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_3_weight, conv5_3_weight,
               sizeof(float) * conv5_3_in_channel * conv5_3_out_channel * conv5_3_kernel_size * conv5_3_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_3_bias, conv5_3_bias, sizeof(float) * conv5_3_out_channel, cudaMemcpyHostToDevice);

    cudaMemcpy(d_fc1_weight, fc1_weight, sizeof(float) * fc1_in_channel * fc1_out_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_bias, fc1_bias, sizeof(float) * fc1_out_channel, cudaMemcpyHostToDevice);

    // copy input image
    size_t image_size = batch * input_size * input_size * input_channel;
    cudaMemcpy(d_image, image, image_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
}

void vgg16_cuda::classify(int *predict, int batch)
{
    // read logits back to cpu
    cudaMemcpy(output, d_output, sizeof(float) * output_size * batch, cudaMemcpyDeviceToHost);
    // Softmax
    softmax(output, predict, batch, output_size);
}

vgg16_cuda::~vgg16_cuda()
{
    cudaFree(d_conv1_1_weight);
    cudaFree(d_conv1_2_weight);
    cudaFree(d_conv2_1_weight);
    cudaFree(d_conv2_2_weight);
    cudaFree(d_conv3_1_weight);
    cudaFree(d_conv3_2_weight);
    cudaFree(d_conv3_3_weight);
    cudaFree(d_conv4_1_weight);
    cudaFree(d_conv4_2_weight);
    cudaFree(d_conv4_3_weight);
    cudaFree(d_conv5_1_weight);
    cudaFree(d_conv5_2_weight);
    cudaFree(d_conv5_3_weight);

    cudaFree(d_conv1_1_bias);
    cudaFree(d_conv1_2_bias);
    cudaFree(d_conv2_1_bias);
    cudaFree(d_conv2_2_bias);
    cudaFree(d_conv3_1_bias);
    cudaFree(d_conv3_2_bias);
    cudaFree(d_conv3_3_bias);
    cudaFree(d_conv4_1_bias);
    cudaFree(d_conv4_2_bias);
    cudaFree(d_conv4_3_bias);
    cudaFree(d_conv5_1_bias);
    cudaFree(d_conv5_2_bias);
    cudaFree(d_conv5_3_bias);

    cudaFree(d_fc1_weight);
    cudaFree(d_fc1_bias);

    cudaFree(d_image);
    cudaFree(d_input);

    cudaFree(d_input_padded);
    cudaFree(d_C1_1_feature_map);
    cudaFree(d_C1_1_feature_map_padded);
    cudaFree(d_C1_2_feature_map);
    cudaFree(d_S1_feature_map);

    cudaFree(d_S1_feature_map_padded);
    cudaFree(d_C2_1_feature_map);
    cudaFree(d_C2_1_feature_map_padded);
    cudaFree(d_C2_2_feature_map);
    cudaFree(d_S2_feature_map);

    cudaFree(d_S2_feature_map_padded);
    cudaFree(d_C3_1_feature_map);
    cudaFree(d_C3_1_feature_map_padded);
    cudaFree(d_C3_2_feature_map);
    cudaFree(d_C3_2_feature_map_padded);
    cudaFree(d_C3_3_feature_map);
    cudaFree(d_S3_feature_map);

    cudaFree(d_S3_feature_map_padded);
    cudaFree(d_C4_1_feature_map);
    cudaFree(d_C4_1_feature_map_padded);
    cudaFree(d_C4_2_feature_map);
    cudaFree(d_C4_2_feature_map_padded);
    cudaFree(d_C4_3_feature_map);
    cudaFree(d_S4_feature_map);

    cudaFree(d_S4_feature_map_padded);
    cudaFree(d_C5_1_feature_map);
    cudaFree(d_C5_1_feature_map_padded);
    cudaFree(d_C5_2_feature_map);
    cudaFree(d_C5_2_feature_map_padded);
    cudaFree(d_C5_3_feature_map);
    cudaFree(d_S5_feature_map);

    cudaFree(d_output);
    cudaFree(d_predict_cuda);
}
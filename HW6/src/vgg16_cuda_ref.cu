#include "vgg16_cuda.h"

__global__ void normalize(const uint8_t* const image, float* input,
                          int N) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float max_int = 255.0L;
    float mean = 0.5L;
    float var = 0.5L;

    if (id < N) input[id] = (image[id] / max_int - mean) / var;
}

__global__ void relu(float* feature_map, int N) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < N) {
        feature_map[id] = (feature_map[id] < 0.0f ? 0.0f : feature_map[id]);
    }
}

__global__ void pad(float* input, float* input_padded,
                    int B, int C,
                    int H, int W, int P) {
    int H_OUT = H + 2 * P;
    int W_OUT = W + 2 * P;

    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = threadIdx.x;
    int w = threadIdx.y;

    // Init values
    int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;

    // Set output with max value
    int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) +
                       (h + P) * W_OUT + (w + P);

    input_padded[output_index] = input[input_base];
}

__global__ void im2col(float* input, float* output, int W, int H, int IC, int OC, int K) {
    int n = blockIdx.x;
    int w = threadIdx.y;
    int h = threadIdx.x;
    int H_OUT = H - K + 1;
    int W_OUT = W - K + 1;

    int K_K = K * K;
    int W_H = W * H;
    int row_dim = W_OUT * W_OUT;
    int col_dim = K_K * IC;
    int N_IC_W_H = n * (IC * W_H);
    int RD_CD_N_H = row_dim * col_dim * n + h;
    int offset = (h / H_OUT) * H + (h % H_OUT);

    for (int c = 0; c < IC; c++) {
        int in_base = N_IC_W_H + c * W_H + offset;
        int out_base = RD_CD_N_H + c * K_K * row_dim;
        for (int hi = 0; hi < K; hi++) {
            for (int wi = 0; wi < K; wi++) {
                output[out_base + (hi * K + wi) * row_dim] = input[in_base + hi * W + wi];
            }
        }
    }
}

__global__ void matmul(float* input, float* output, float* weight, float* bias,
                       int H, int W,
                       int IC, int OC, int K) {
    int b = blockIdx.z;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    int k = IC * K * K;
    int w = H * H;
    int h = OC;
    int in_base = b * w * k;
    int out_base = b * w * h;

    if (r < h && c < w) {
        float sum = bias[r];
        for (int ki = 0; ki < k; ki++) {
            sum += weight[r * k + ki] * input[in_base + ki * w + c];
        }
        output[out_base + r * w + c] = sum;
    }
}

#define TILE_WIDTH 16
__global__ void matmul1(float* input, float* output, float* weight, float* bias,
                        int H, int W,
                        int IC, int OC, int K) {
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    int b = blockIdx.z;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int w = IC * H * H;
    int h = OC;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    if (r < h && c < w) {
        float Pvalue = 0;
        for (int m = 0; m < W / TILE_WIDTH; m++) {
            // Load shared memory phase
            subTileM[ty][tx] = input[Row * W + m * TILE_WIDTH + tx];
            subTileN[ty][tx] = weight[Row * W + m * TILE_WIDTH + tx];
            __syncthreads();  // Wait for the copy to finish // Compute phase
            for (int k = 0; k < TILE_WIDTH; ++k)
                Pvalue += subTileM[ty][k] * subTileN[k][tx];
            __syncthreads();  // Wait for the compute to be complete
            // Loop over the M and N tiles required to compute the P element
        }
        output[Row * W + Col] = Pvalue;
    }
}

__global__ void conv(float* input, float* output, float* weight, float* bias,
                     int H, int W,
                     int IC, int OC, int K) {
    int H_OUT = H - (K - 1);
    int W_OUT = W - (K - 1);

    int b = blockIdx.x;
    int oc = blockIdx.y;
    int h = threadIdx.x;
    int w = threadIdx.y;

    int H_W = H * W;
    int K_K = K * K;
    int b_IC_H_W = b * (IC * H_W);
    int oc_IC_K_K = oc * (IC * K_K);

    float sum = bias[oc];
    for (int ic = 0; ic < IC; ic++) {
        int input_base = b_IC_H_W + ic * (H_W) + h * (W) + w;
        int kernel_base = oc_IC_K_K + ic * (K_K);
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                sum += input[input_base + kh * W + kw] * weight[kernel_base + kh * K + kw];
            }
        }
    }

    int output_index = b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
    output[output_index] = sum;
}

__global__ void pool(float* input, float* output,
                     int B, int C,
                     int H, int W) {
    int scale = 2;

    int b = blockIdx.x;
    int c = blockIdx.y;
    int w = threadIdx.x;
    int h = threadIdx.y;
    int H_OUT = H / scale;
    int W_OUT = W / scale;
    int r_w = w * 2;
    int r_h = h * 2;

    int input_base = b * (C * H * W) + c * (H * W);

    float max_val = 0.0;
    for (int kh = 0; kh < 2; kh++) {
        for (int kw = 0; kw < 2; kw++) {
            int idx = input_base + (r_h + kh) * W + r_w + kw;
            max_val = (input[idx] > max_val) ? input[idx] : max_val;
        }
    }
    output[b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) + h * W_OUT + w] = max_val;
}

__global__ void fc(float* input, float* output, float* weight, float* bias,
                   int B, int IC, int OC) {
    int b = blockIdx.x;
    int oc = threadIdx.x;

    float sum = bias[oc];
    for (int ic = 0; ic < IC; ic++) {
        sum += input[b * IC + ic] * weight[oc * IC + ic];
    }
    output[b * OC + oc] = sum;
}

void vgg16_cuda::predict(int batch) {
    // the num of Threads
    const float THREAD = 256.0f;
    const float DIV = 16.0f;
    const int MUL = 16;

    // Nomalize
    dim3 DimGrid_normal(ceil(batch * input_channel * input_size * input_size / THREAD), 1, 1);
    dim3 DimBlock_normal(THREAD, 1, 1);
    normalize<<<DimGrid_normal, DimBlock_normal>>>(d_image, d_input,
                                                   batch * input_channel * input_size * input_size);

    //////////BLOCK 1/////////////////////////////////
    // TODO: Implement pad
    dim3 DimGrid_pad_1_1(batch, input_channel, 1);
    dim3 DimBlock_pad_1_1(input_size, input_size, 1);
    pad<<<DimGrid_pad_1_1, DimBlock_pad_1_1>>>(d_input, d_input_padded,
                                               batch, input_channel,
                                               input_size, input_size, conv1_1_padding_size);

    // TODO: Implement conv1_1
    // dim3 DimGrid_conv_1_1(batch, C1_1_channel, 1);
    // dim3 DimBlock_conv_1_1(C1_1_size, C1_1_size, 1);
    // conv<<<DimGrid_conv_1_1, DimBlock_conv_1_1>>>(d_input_padded, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias,
    //                                     input_size + 2 * conv1_1_padding_size, input_size + 2 * conv1_1_padding_size,
    //                                     conv1_1_in_channel, conv1_1_out_channel, conv1_1_kernel_size);

    dim3 DimGrid_im_1_1(batch, 1, 1);
    dim3 DimBlock_im1_1(C1_1_size * C1_1_size, 1, 1);
    im2col<<<DimGrid_im_1_1, DimBlock_im1_1>>>(d_input_padded, d_input_padded_im2col,
                                               input_size + 2 * conv1_1_padding_size, input_size + 2 * conv1_1_padding_size,
                                               conv1_1_in_channel, conv1_1_out_channel, conv1_1_kernel_size);

    dim3 DimGrid_im2col_1_1(ceil((C1_1_size * C1_1_size) / DIV), ceil(C1_1_channel / DIV), batch);
    dim3 DimBlock_im2col_1_1(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_1_1, DimBlock_im2col_1_1>>>(d_input_padded_im2col, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias,
                                                        C1_1_size, C1_1_size,
                                                        conv1_1_in_channel, conv1_1_out_channel, conv1_1_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C1_1_channel * C1_1_size * C1_1_size / THREAD), THREAD>>>(d_C1_1_feature_map,
                                                                                  batch * C1_1_channel * C1_1_size * C1_1_size);

    // TODO: Implement pad
    dim3 DimGrid_pad_1_2(batch, C1_1_channel, 1);
    dim3 DimBlock_pad_1_2(C1_1_size, C1_1_size, 1);
    pad<<<DimGrid_pad_1_2, DimBlock_pad_1_2>>>(d_C1_1_feature_map, d_C1_1_feature_map_padded,
                                               batch, C1_1_channel,
                                               C1_1_size, C1_1_size, conv1_2_padding_size);

    // TODO: Implement conv1_2
    // dim3 DimGrid_conv_1_2(batch, C1_2_channel, 1);
    // dim3 DimBlock_conv_1_2(C1_2_size, C1_2_size, 1);
    // conv<<<DimGrid_conv_1_2, DimBlock_conv_1_2>>>(d_C1_1_feature_map_padded, d_C1_2_feature_map, d_conv1_2_weight, d_conv1_2_bias,
    //                                     C1_1_size + 2 * conv1_2_padding_size, C1_1_size + 2 * conv1_2_padding_size,
    //                                     conv1_2_in_channel, conv1_2_out_channel, conv1_2_kernel_size);

    dim3 DimGrid_im_1_2(batch, 1, 1);
    dim3 DimBlock_im1_2(C1_2_size * C1_2_size, 1, 1);
    im2col<<<DimGrid_im_1_2, DimBlock_im1_2>>>(d_C1_1_feature_map_padded, d_C1_1_feature_map_padded_im2col,
                                               C1_1_size + 2 * conv1_2_padding_size, C1_1_size + 2 * conv1_2_padding_size,
                                               conv1_2_in_channel, conv1_2_out_channel, conv1_2_kernel_size);

    dim3 DimGrid_im2col_1_2(ceil((C1_2_size * C1_2_size) / DIV), ceil(C1_2_channel / DIV), batch);
    dim3 DimBlock_im2col_1_2(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_1_2, DimBlock_im2col_1_2>>>(d_C1_1_feature_map_padded_im2col, d_C1_2_feature_map, d_conv1_2_weight, d_conv1_2_bias,
                                                        C1_2_size, C1_2_size,
                                                        conv1_2_in_channel, conv1_2_out_channel, conv1_2_kernel_size);

    // // TODO: Implement relu
    relu<<<ceil(batch * C1_2_channel * C1_2_size * C1_2_size / THREAD), THREAD>>>(d_C1_2_feature_map,
                                                                                  batch * C1_2_channel * C1_2_size * C1_2_size);

    // TODO: Implement pool
    dim3 DimGrid_pl_1(batch, S1_channel, 1);
    dim3 DimBlock_pl_1(S1_size, S1_size, 1);
    pool<<<DimGrid_pl_1, DimBlock_pl_1>>>(d_C1_2_feature_map, d_S1_feature_map,
                                          batch, C1_2_channel,
                                          C1_2_size, C1_2_size);

    //////////BLOCK 2/////////////////////////////////
    // TODO: Implement pad
    dim3 DimGrid_pad_2_1(batch, S1_channel, 1);
    dim3 DimBlock_pad_2_1(S1_size, S1_size, 1);
    pad<<<DimGrid_pad_2_1, DimBlock_pad_2_1>>>(d_S1_feature_map, d_S1_feature_map_padded,
                                               batch, S1_channel,
                                               S1_size, S1_size, conv2_1_padding_size);

    // TODO: Implement conv2_1
    // dim3 DimGrid_conv_2_1(batch, C2_1_channel, 1);
    // dim3 DimBlock_conv_2_1(C2_1_size, C2_1_size, 1);
    // conv<<<DimGrid_conv_2_1, DimBlock_conv_2_1>>>(d_S1_feature_map_padded, d_C2_1_feature_map, d_conv2_1_weight, d_conv2_1_bias,
    //                                     S1_size + 2 * conv2_1_padding_size, S1_size + 2 * conv2_1_padding_size,
    //                                     conv2_1_in_channel, conv2_1_out_channel, conv2_1_kernel_size);

    dim3 DimGrid_im_2_1(batch, 1, 1);
    dim3 DimBlock_im2_1(C2_1_size * C2_1_size, 1, 1);
    im2col<<<DimGrid_im_2_1, DimBlock_im2_1>>>(d_S1_feature_map_padded, d_S1_feature_map_padded_im2col,
                                               S1_size + 2 * conv2_1_padding_size, S1_size + 2 * conv2_1_padding_size,
                                               conv2_1_in_channel, conv2_1_out_channel, conv2_1_kernel_size);

    dim3 DimGrid_im2col_2_1(ceil((C2_1_size * C2_1_size) / DIV), ceil(C2_1_channel / DIV), batch);
    dim3 DimBlock_im2col_2_1(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_2_1, DimBlock_im2col_2_1>>>(d_S1_feature_map_padded_im2col, d_C2_1_feature_map, d_conv2_1_weight, d_conv2_1_bias,
                                                        C2_1_size, C2_1_size,
                                                        conv2_1_in_channel, conv2_1_out_channel, conv2_1_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C2_1_channel * C2_1_size * C2_1_size / THREAD), THREAD>>>(d_C2_1_feature_map,
                                                                                  batch * C2_1_channel * C2_1_size * C2_1_size);

    // TODO: Implement pad
    dim3 DimGrid_pad_2_2(batch, C2_1_channel, 1);
    dim3 DimBlock_pad_2_2(C2_1_size, C2_1_size, 1);
    pad<<<DimGrid_pad_2_2, DimBlock_pad_2_2>>>(d_C2_1_feature_map, d_C2_1_feature_map_padded,
                                               batch, C2_1_channel,
                                               C2_1_size, C2_1_size, conv2_2_padding_size);

    // TODO: Implement conv2_2
    // dim3 DimGrid_conv_2_2(batch, C2_2_channel, 1);
    // dim3 DimBlock_conv_2_2(C2_2_size, C2_2_size, 1);
    // conv<<<DimGrid_conv_2_2, DimBlock_conv_2_2>>>(d_C2_1_feature_map_padded, d_C2_2_feature_map, d_conv2_2_weight, d_conv2_2_bias,
    //                                     C2_1_size + 2 * conv2_2_padding_size, C2_1_size + 2 * conv2_2_padding_size,
    //                                      conv2_2_in_channel, conv2_2_out_channel, conv2_2_kernel_size);

    dim3 DimGrid_im_2_2(batch, 1, 1);
    dim3 DimBlock_im2_2(C2_2_size * C2_2_size, 1, 1);
    im2col<<<DimGrid_im_2_2, DimBlock_im2_2>>>(d_C2_1_feature_map_padded, d_C2_1_feature_map_padded_im2col,
                                               C2_1_size + 2 * conv2_2_padding_size, C2_1_size + 2 * conv2_2_padding_size,
                                               conv2_2_in_channel, conv2_2_out_channel, conv2_2_kernel_size);

    dim3 DimGrid_im2col_2_2(ceil((C2_2_size * C2_2_size) / DIV), ceil(C2_2_channel / DIV), batch);
    dim3 DimBlock_im2col_2_2(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_2_2, DimBlock_im2col_2_2>>>(d_C2_1_feature_map_padded_im2col, d_C2_2_feature_map, d_conv2_2_weight, d_conv2_2_bias,
                                                        C2_2_size, C2_2_size,
                                                        conv2_2_in_channel, conv2_2_out_channel, conv2_2_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C2_2_channel * C2_2_size * C2_2_size / THREAD), THREAD>>>(d_C2_2_feature_map,
                                                                                  batch * C2_2_channel * C2_2_size * C2_2_size);

    // TODO: Implement pool
    dim3 DimGrid_pl_2(batch, S2_channel, 1);
    dim3 DimBlock_pl_2(S2_size, S2_size, 1);
    pool<<<DimGrid_pl_2, DimBlock_pl_2>>>(d_C2_2_feature_map, d_S2_feature_map,
                                          batch, C2_2_channel,
                                          C2_2_size, C2_2_size);

    ////////BLOCK 3/////////////////////////////////
    // TODO: Implement pad
    dim3 DimGrid_pad_3_1(batch, S2_channel, 1);
    dim3 DimBlock_pad_3_1(S2_size, S2_size, 1);
    pad<<<DimGrid_pad_3_1, DimBlock_pad_3_1>>>(d_S2_feature_map, d_S2_feature_map_padded,
                                               batch, S2_channel,
                                               S2_size, S2_size, conv3_1_padding_size);

    // TODO: Implement c1
    // dim3 DimGrid_conv_3_1(batch, C3_1_channel, 1);
    // dim3 DimBlock_conv_3_1(C3_1_size, C3_1_size, 1);
    // conv<<<DimGrid_conv_3_1, DimBlock_conv_3_1>>>(d_S2_feature_map_padded, d_C3_1_feature_map, d_conv3_1_weight, d_conv3_1_bias,
    //                                     S2_size + 2 * conv3_1_padding_size, S2_size + 2 * conv3_1_padding_size,
    //                                     conv3_1_in_channel, conv3_1_out_channel, conv3_1_kernel_size);

    dim3 DimGrid_im_3_1(batch, 1, 1);
    dim3 DimBlock_im3_1(C3_1_size * C3_1_size, 1, 1);
    im2col<<<DimGrid_im_3_1, DimBlock_im3_1>>>(d_S2_feature_map_padded, d_S2_feature_map_padded_im2col,
                                               S2_size + 2 * conv3_1_padding_size, S2_size + 2 * conv3_1_padding_size,
                                               conv3_1_in_channel, conv3_1_out_channel, conv3_1_kernel_size);

    dim3 DimGrid_im2col_3_1(ceil((C3_1_size * C3_1_size) / DIV), ceil(C3_1_channel / DIV), batch);
    dim3 DimBlock_im2col_3_1(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_3_1, DimBlock_im2col_3_1>>>(d_S2_feature_map_padded_im2col, d_C3_1_feature_map, d_conv3_1_weight, d_conv3_1_bias,
                                                        C3_1_size, C3_1_size,
                                                        conv3_1_in_channel, conv3_1_out_channel, conv3_1_kernel_size);

    // TODO: Implement reluonv3_
    relu<<<ceil(batch * C3_1_channel * C3_1_size * C3_1_size / THREAD), THREAD>>>(d_C3_1_feature_map,
                                                                                  batch * C3_1_channel * C3_1_size * C3_1_size);

    // TODO: Implement pad
    dim3 DimGrid_pad_3_2(batch, C3_1_channel, 1);
    dim3 DimBlock_pad_3_2(C3_1_size, C3_1_size, 1);
    pad<<<DimGrid_pad_3_2, DimBlock_pad_3_2>>>(d_C3_1_feature_map, d_C3_1_feature_map_padded,
                                               batch, C3_1_channel,
                                               C3_1_size, C3_1_size, conv3_2_padding_size);

    // TODO: Implement conv3_2
    // dim3 DimGrid_conv_3_2(batch, C3_2_channel, 1);
    // dim3 DimBlock_conv_3_2(C3_2_size, C3_2_size, 1);
    // conv<<<DimGrid_conv_3_2, DimBlock_conv_3_2>>>(d_C3_1_feature_map_padded, d_C3_2_feature_map, d_conv3_2_weight, d_conv3_2_bias,
    //                                     C3_1_size + 2 * conv3_2_padding_size, C3_1_size + 2 * conv3_2_padding_size,
    //                                     conv3_2_in_channel, conv3_2_out_channel, conv3_2_kernel_size);

    dim3 DimGrid_im_3_2(batch, 1, 1);
    dim3 DimBlock_im3_2(C3_2_size * C3_2_size, 1, 1);
    im2col<<<DimGrid_im_3_2, DimBlock_im3_2>>>(d_C3_1_feature_map_padded, d_C3_1_feature_map_padded_im2col,
                                               C3_1_size + 2 * conv3_2_padding_size, C3_1_size + 2 * conv3_2_padding_size,
                                               conv3_2_in_channel, conv3_2_out_channel, conv3_2_kernel_size);

    dim3 DimGrid_im2col_3_2(ceil((C3_2_size * C3_2_size) / DIV), ceil(C3_2_channel / DIV), batch);
    dim3 DimBlock_im2col_3_2(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_3_2, DimBlock_im2col_3_2>>>(d_C3_1_feature_map_padded_im2col, d_C3_2_feature_map, d_conv3_2_weight, d_conv3_2_bias,
                                                        C3_2_size, C3_2_size,
                                                        conv3_2_in_channel, conv3_2_out_channel, conv3_2_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C3_2_channel * C3_2_size * C3_2_size / THREAD), THREAD>>>(d_C3_2_feature_map,
                                                                                  batch * C3_2_channel * C3_2_size * C3_2_size);

    // TODO: Implement pad
    dim3 DimGrid_pad_3_3(batch, C3_2_channel, 1);
    dim3 DimBlock_pad_3_3(C3_2_size, C3_2_size, 1);
    pad<<<DimGrid_pad_3_3, DimBlock_pad_3_3>>>(d_C3_2_feature_map, d_C3_2_feature_map_padded,
                                               batch, C3_2_channel,
                                               C3_2_size, C3_2_size, conv3_3_padding_size);

    // TODO: Implement conv3_3
    // dim3 DimGrid_conv_3_3(batch, C3_3_channel, 1);
    // dim3 DimBlock_conv_3_3(C3_3_size, C3_3_size, 1);
    // conv<<<DimGrid_conv_3_3, DimBlock_conv_3_3>>>(d_C3_2_feature_map_padded, d_C3_3_feature_map, d_conv3_3_weight, d_conv3_3_bias,
    //                                     C3_3_size + 2 * conv3_3_padding_size, C3_3_size + 2 * conv3_3_padding_size,
    //                                     conv3_3_in_channel, conv3_3_out_channel, conv3_3_kernel_size);

    dim3 DimGrid_im_3_3(batch, 1, 1);
    dim3 DimBlock_im3_3(C3_3_size * C3_3_size, 1, 1);
    im2col<<<DimGrid_im_3_3, DimBlock_im3_3>>>(d_C3_2_feature_map_padded, d_C3_2_feature_map_padded_im2col,
                                               C3_2_size + 2 * conv3_3_padding_size, C3_2_size + 2 * conv3_3_padding_size,
                                               conv3_3_in_channel, conv3_3_out_channel, conv3_3_kernel_size);
    dim3 DimGrid_im2col_3_3(ceil((C3_3_size * C3_3_size) / DIV), ceil(C3_3_channel / DIV), batch);
    dim3 DimBlock_im2col_3_3(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_3_3, DimBlock_im2col_3_3>>>(d_C3_2_feature_map_padded_im2col, d_C3_3_feature_map, d_conv3_3_weight, d_conv3_3_bias,
                                                        C3_3_size, C3_3_size,
                                                        conv3_3_in_channel, conv3_3_out_channel, conv3_3_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C3_3_channel * C3_3_size * C3_3_size / THREAD), THREAD>>>(d_C3_3_feature_map,
                                                                                  batch * C3_3_channel * C3_3_size * C3_3_size);

    // TODO: Implement pool
    dim3 DimGrid_pl_3(batch, S3_channel, 1);
    dim3 DimBlock_pl_3(S3_size, S3_size, 1);
    pool<<<DimGrid_pl_3, DimBlock_pl_3>>>(d_C3_3_feature_map, d_S3_feature_map,
                                          batch, C3_3_channel,
                                          C3_3_size, C3_3_size);

    //////////BLOCK 4/////////////////////////////////
    // TODO: Implement pad
    dim3 DimGrid_pad_4_1(batch, S3_channel, 1);
    dim3 DimBlock_pad_4_1(S3_size, S3_size, 1);
    pad<<<DimGrid_pad_4_1, DimBlock_pad_4_1>>>(d_S3_feature_map, d_S3_feature_map_padded,
                                               batch, S3_channel,
                                               S3_size, S3_size, conv4_1_padding_size);

    // TODO: Implement conv4_1
    // dim3 DimGrid_conv_4_1(batch, C4_1_channel, 1);
    // dim3 DimBlock_conv_4_1(C4_1_size, C4_1_size, 1);
    // conv<<<DimGrid_conv_4_1, DimBlock_conv_4_1>>>(d_S3_feature_map_padded, d_C4_1_feature_map, d_conv4_1_weight, d_conv4_1_bias,
    //                                     S3_size + 2 * conv4_1_padding_size, S3_size + 2 * conv4_1_padding_size,
    //                                     conv4_1_in_channel, conv4_1_out_channel, conv4_1_kernel_size);

    dim3 DimGrid_im_4_1(batch, 1, 1);
    dim3 DimBlock_im4_1(C4_1_size * C4_1_size, 1, 1);
    im2col<<<DimGrid_im_4_1, DimBlock_im4_1>>>(d_S3_feature_map_padded, d_S3_feature_map_padded_im2col,
                                               S3_size + 2 * conv4_1_padding_size, S3_size + 2 * conv4_1_padding_size,
                                               conv4_1_in_channel, conv4_1_out_channel, conv4_1_kernel_size);

    dim3 DimGrid_im2col_4_1(ceil((C4_1_size * C4_1_size) / DIV), ceil(C4_1_channel / DIV), batch);
    dim3 DimBlock_im2col_4_1(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_4_1, DimBlock_im2col_4_1>>>(d_S3_feature_map_padded_im2col, d_C4_1_feature_map, d_conv4_1_weight, d_conv4_1_bias,
                                                        C4_1_size, C4_1_size,
                                                        conv4_1_in_channel, conv4_1_out_channel, conv4_1_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C4_1_channel * C4_1_size * C4_1_size / THREAD), THREAD>>>(d_C4_1_feature_map,
                                                                                  batch * C4_1_channel * C4_1_size * C4_1_size);

    // TODO: Implement pad
    dim3 DimGrid_pad_4_2(batch, C4_1_channel, 1);
    dim3 DimBlock_pad_4_2(C4_1_size, C4_1_size, 1);
    pad<<<DimGrid_pad_4_2, DimBlock_pad_4_2>>>(d_C4_1_feature_map, d_C4_1_feature_map_padded,
                                               batch, C4_1_channel,
                                               C4_1_size, C4_1_size, conv4_2_padding_size);

    // TODO: Implement conv4_2
    // dim3 DimGrid_conv_4_2(batch, C4_2_channel, 1);
    // dim3 DimBlock_conv_4_2(C4_2_size, C4_2_size, 1);
    // conv<<<DimGrid_conv_4_2, DimBlock_conv_4_2>>>(d_C4_1_feature_map_padded, d_C4_2_feature_map, d_conv4_2_weight, d_conv4_2_bias,
    //                                     C4_1_size + 2 * conv4_2_padding_size, C4_1_size + 2 * conv4_2_padding_size,
    //                                     conv4_2_in_channel, conv4_2_out_channel, conv4_2_kernel_size);

    dim3 DimGrid_im_4_2(batch, 1, 1);
    dim3 DimBlock_im4_2(C4_2_size * C4_2_size, 1, 1);
    im2col<<<DimGrid_im_4_2, DimBlock_im4_2>>>(d_C4_1_feature_map_padded, d_C4_1_feature_map_padded_im2col,
                                               C4_1_size + 2 * conv4_2_padding_size, C4_1_size + 2 * conv4_2_padding_size,
                                               conv4_2_in_channel, conv4_2_out_channel, conv4_2_kernel_size);
    dim3 DimGrid_im2col_4_2(ceil((C4_2_size * C4_2_size) / DIV), ceil(C4_2_channel / DIV), batch);
    dim3 DimBlock_im2col_4_2(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_4_2, DimBlock_im2col_4_2>>>(d_C4_1_feature_map_padded_im2col, d_C4_2_feature_map, d_conv4_2_weight, d_conv4_2_bias,
                                                        C4_2_size, C4_2_size,
                                                        conv4_2_in_channel, conv4_2_out_channel, conv4_2_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C4_2_channel * C4_2_size * C4_2_size / THREAD), THREAD>>>(d_C4_2_feature_map,
                                                                                  batch * C4_2_channel * C4_2_size * C4_2_size);

    // TODO: Implement pad
    dim3 DimGrid_pad_4_3(batch, C4_2_channel, 1);
    dim3 DimBlock_pad_4_3(C4_2_size, C4_2_size, 1);
    pad<<<DimGrid_pad_4_3, DimBlock_pad_4_3>>>(d_C4_2_feature_map, d_C4_2_feature_map_padded, batch, C4_2_channel, C4_2_size, C4_2_size, conv4_3_padding_size);

    // TODO: Implement conv4_3
    // dim3 DimGrid_conv_4_3(batch, C4_3_channel, 1);
    // dim3 DimBlock_conv_4_3(C4_3_size, C4_3_size, 1);
    // conv<<<DimGrid_conv_4_3, DimBlock_conv_4_3>>>(d_C4_2_feature_map_padded, d_C4_3_feature_map, d_conv4_3_weight, d_conv4_3_bias,
    //                                     C4_3_size + 2 * conv4_3_padding_size, C4_3_size + 2 * conv4_3_padding_size,
    //                                     conv4_3_in_channel, conv4_3_out_channel, conv4_3_kernel_size);

    dim3 DimGrid_im_4_3(batch, 1, 1);
    dim3 DimBlock_im4_3(C4_3_size * C4_3_size, 1, 1);
    im2col<<<DimGrid_im_4_3, DimBlock_im4_3>>>(d_C4_2_feature_map_padded, d_C4_2_feature_map_padded_im2col,
                                               C4_2_size + 2 * conv4_3_padding_size, C4_2_size + 2 * conv4_3_padding_size,
                                               conv4_3_in_channel, conv4_3_out_channel, conv4_3_kernel_size);
    dim3 DimGrid_im2col_4_3(ceil((C4_3_size * C4_3_size) / DIV), ceil(C4_3_channel / DIV), batch);
    dim3 DimBlock_im2col_4_3(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_4_3, DimBlock_im2col_4_3>>>(d_C4_2_feature_map_padded_im2col, d_C4_3_feature_map, d_conv4_3_weight, d_conv4_3_bias,
                                                        C4_3_size, C4_3_size,
                                                        conv4_3_in_channel, conv4_3_out_channel, conv4_3_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C4_3_channel * C4_3_size * C4_3_size / THREAD), THREAD>>>(d_C4_3_feature_map, batch * C4_3_channel * C4_3_size * C4_3_size);
    // TODO: Implement pool
    dim3 DimGrid_pl_4(batch, S4_channel, 1);
    dim3 DimBlock_pl_4(S4_size, S4_size, 1);
    pool<<<DimGrid_pl_4, DimBlock_pl_4>>>(d_C4_3_feature_map, d_S4_feature_map, batch, S4_channel, C4_3_size, C4_3_size);

    //////////BLOCK 5/////////////////////////////////
    // // TODO: Implement pad
    dim3 DimGrid_pad_5_1(batch, S4_channel, 1);
    dim3 DimBlock_pad_5_1(S4_size, S4_size, 1);
    pad<<<DimGrid_pad_5_1, DimBlock_pad_5_1>>>(d_S4_feature_map, d_S4_feature_map_padded, batch, S4_channel, S4_size, S4_size, conv5_1_padding_size);

    // TODO: Implement conv5_1
    // dim3 DimGrid_conv_5_1(batch, C5_1_channel, 1);
    // dim3 DimBlock_conv_5_1(C5_1_size, C5_1_size, 1);
    // conv<<<DimGrid_conv_5_1, DimBlock_conv_5_1>>>(d_S4_feature_map_padded, d_C5_1_feature_map, d_conv5_1_weight, d_conv5_1_bias,
    //                                     S4_size + 2 * conv5_1_padding_size, S4_size + 2 * conv5_1_padding_size,
    //                                     conv5_1_in_channel, conv5_1_out_channel, conv5_1_kernel_size);

    dim3 DimGrid_im_5_1(batch, 1, 1);
    dim3 DimBlock_im5_1(C5_1_size * C5_1_size, 1, 1);
    im2col<<<DimGrid_im_5_1, DimBlock_im5_1>>>(d_S4_feature_map_padded, d_S4_feature_map_padded_im2col,
                                               S4_size + 2 * conv5_1_padding_size, S4_size + 2 * conv5_1_padding_size,
                                               conv5_1_in_channel, conv4_1_out_channel, conv5_1_kernel_size);

    dim3 DimGrid_im2col_5_1(ceil((C5_1_size * C5_1_size) / DIV), ceil(C5_1_channel / DIV), batch);
    dim3 DimBlock_im2col_5_1(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_5_1, DimBlock_im2col_5_1>>>(d_S4_feature_map_padded_im2col, d_C5_1_feature_map, d_conv5_1_weight, d_conv5_1_bias,
                                                        C5_1_size, C5_1_size,
                                                        conv5_1_in_channel, conv5_1_out_channel, conv5_1_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C5_1_channel * C5_1_size * C5_1_size / THREAD), THREAD>>>(d_C5_1_feature_map, batch * C5_1_channel * C5_1_size * C5_1_size);

    // TODO: Implement pad
    dim3 DimGrid_pad_5_2(batch, C5_1_channel, 1);
    dim3 DimBlock_pad_5_2(C5_1_size, C5_1_size, 1);
    pad<<<DimGrid_pad_5_2, DimBlock_pad_5_2>>>(d_C5_1_feature_map, d_C5_1_feature_map_padded, batch, C5_1_channel, C5_1_size, C5_1_size, conv5_2_padding_size);

    // TODO: Implement conv5_2
    // dim3 DimGrid_conv_5_2(batch, C5_2_channel, 1);
    // dim3 DimBlock_conv_5_2(C5_2_size, C5_2_size, 1);
    // conv<<<DimGrid_conv_5_2, DimBlock_conv_5_2>>>(d_C5_1_feature_map_padded, d_C5_2_feature_map, d_conv5_2_weight, d_conv5_2_bias,
    //                                     C5_1_size + 2 * conv5_2_padding_size, C5_1_size + 2 * conv5_2_padding_size,
    //                                     conv5_2_in_channel, conv5_2_out_channel, conv5_2_kernel_size);

    dim3 DimGrid_im_5_2(batch, 1, 1);
    dim3 DimBlock_im5_2(C5_2_size * C5_2_size, 1, 1);
    im2col<<<DimGrid_im_5_2, DimBlock_im5_2>>>(d_C5_1_feature_map_padded, d_C5_1_feature_map_padded_im2col,
                                               C5_1_size + 2 * conv5_2_padding_size, C5_1_size + 2 * conv5_2_padding_size,
                                               conv5_2_in_channel, conv5_2_out_channel, conv5_2_kernel_size);
    dim3 DimGrid_im2col_5_2(ceil((C5_2_size * C5_2_size) / DIV), ceil(C5_2_channel / DIV), batch);
    dim3 DimBlock_im2col_5_2(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_5_2, DimBlock_im2col_5_2>>>(d_C5_1_feature_map_padded_im2col, d_C5_2_feature_map, d_conv5_2_weight, d_conv5_2_bias,
                                                        C5_2_size, C5_2_size,
                                                        conv5_2_in_channel, conv5_2_out_channel, conv5_2_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C5_2_channel * C5_2_size * C5_2_size / THREAD), THREAD>>>(d_C5_2_feature_map, batch * C5_2_channel * C5_2_size * C5_2_size);

    // TODO: Implement pad
    dim3 DimGrid_pad_5_3(batch, C5_2_channel, 1);
    dim3 DimBlock_pad_5_3(C5_2_size, C5_2_size, 1);
    pad<<<DimGrid_pad_5_3, DimBlock_pad_5_3>>>(d_C5_2_feature_map, d_C5_2_feature_map_padded, batch, C5_2_channel, C5_2_size, C5_2_size, conv5_3_padding_size);

    // TODO: Implement conv5_3
    // dim3 DimGrid_conv_5_3(batch, C5_3_channel, 1);
    // dim3 DimBlock_conv_5_3(C5_3_size, C5_3_size, 1);
    // conv<<<DimGrid_conv_5_3, DimBlock_conv_5_3>>>(d_C5_2_feature_map_padded, d_C5_3_feature_map, d_conv5_3_weight, d_conv5_3_bias,
    //                                     C5_3_size + 2 * conv5_3_padding_size, C5_3_size + 2 * conv5_3_padding_size,
    //                                     conv5_3_in_channel, conv5_3_out_channel, conv5_3_kernel_size);

    dim3 DimGrid_im_5_3(batch, 1, 1);
    dim3 DimBlock_im5_3(C5_3_size * C5_3_size, 1, 1);
    im2col<<<DimGrid_im_5_3, DimBlock_im5_3>>>(d_C5_2_feature_map_padded, d_C5_2_feature_map_padded_im2col,
                                               C5_2_size + 2 * conv5_3_padding_size, C5_2_size + 2 * conv5_3_padding_size,
                                               conv5_3_in_channel, conv5_3_out_channel, conv5_3_kernel_size);
    dim3 DimGrid_im2col_5_3(ceil((C5_3_size * C5_3_size) / DIV), ceil(C5_3_channel / DIV), batch);
    dim3 DimBlock_im2col_5_3(MUL, MUL, 1);
    matmul<<<DimGrid_im2col_5_3, DimBlock_im2col_5_3>>>(d_C5_2_feature_map_padded_im2col, d_C5_3_feature_map, d_conv5_3_weight, d_conv5_3_bias,
                                                        C5_3_size, C5_3_size,
                                                        conv5_3_in_channel, conv5_3_out_channel, conv5_3_kernel_size);

    // TODO: Implement relu
    relu<<<ceil(batch * C5_3_channel * C5_3_size * C5_3_size / THREAD), THREAD>>>(d_C5_3_feature_map, batch * C5_3_channel * C5_3_size * C5_3_size);

    // TODO: Implement pool
    dim3 DimGrid_pl_5(batch, S5_channel, 1);
    dim3 DimBlock_pl_5(S5_size, S5_size, 1);
    pool<<<DimGrid_pl_5, DimBlock_pl_5>>>(d_C5_3_feature_map, d_S5_feature_map, batch, C5_3_channel, C5_3_size, C5_3_size);

    //////////////////////////////////////
    // TODO: Implement fc1
    fc<<<batch, fc1_out_channel>>>(d_S5_feature_map, d_output, d_fc1_weight, d_fc1_bias,
                                   batch, fc1_in_channel, fc1_out_channel);

    /* NOTE: unless you want to make a major change to this class structure,
     *  you need to write your output to the device memory d_output
     *  so that classify() can handle the rest.
     */
}

void vgg16_cuda::prepare_device_memory(uint8_t* image) {
    // Alloc Model Parameters

    //////////BLOCK 1/////////////////////////////////
    cudaMalloc((void**)&d_conv1_1_weight,
               sizeof(float) * conv1_1_in_channel * conv1_1_out_channel *
                   conv1_1_kernel_size * conv1_1_kernel_size);
    cudaMalloc((void**)&d_conv1_1_bias, sizeof(float) * conv1_1_out_channel);
    cudaMalloc((void**)&d_conv1_2_weight,
               sizeof(float) * conv1_2_in_channel * conv1_2_out_channel *
                   conv1_2_kernel_size * conv1_2_kernel_size);
    cudaMalloc((void**)&d_conv1_2_bias, sizeof(float) * conv1_2_out_channel);

    //////////BLOCK 2/////////////////////////////////
    cudaMalloc((void**)&d_conv2_1_weight,
               sizeof(float) * conv2_1_in_channel * conv2_1_out_channel *
                   conv2_1_kernel_size * conv2_1_kernel_size);
    cudaMalloc((void**)&d_conv2_1_bias, sizeof(float) * conv2_1_out_channel);
    cudaMalloc((void**)&d_conv2_2_weight,
               sizeof(float) * conv2_2_in_channel * conv2_2_out_channel *
                   conv2_2_kernel_size * conv2_2_kernel_size);
    cudaMalloc((void**)&d_conv2_2_bias, sizeof(float) * conv2_2_out_channel);

    //////////BLOCK 3/////////////////////////////////
    cudaMalloc((void**)&d_conv3_1_weight,
               sizeof(float) * conv3_1_in_channel * conv3_1_out_channel *
                   conv3_1_kernel_size * conv3_1_kernel_size);
    cudaMalloc((void**)&d_conv3_1_bias, sizeof(float) * conv3_1_out_channel);
    cudaMalloc((void**)&d_conv3_2_weight,
               sizeof(float) * conv3_2_in_channel * conv3_2_out_channel *
                   conv3_2_kernel_size * conv3_2_kernel_size);
    cudaMalloc((void**)&d_conv3_2_bias, sizeof(float) * conv3_2_out_channel);
    cudaMalloc((void**)&d_conv3_3_weight,
               sizeof(float) * conv3_3_in_channel * conv3_3_out_channel *
                   conv3_3_kernel_size * conv3_3_kernel_size);
    cudaMalloc((void**)&d_conv3_3_bias, sizeof(float) * conv3_3_out_channel);

    //////////BLOCK 4/////////////////////////////////
    cudaMalloc((void**)&d_conv4_1_weight,
               sizeof(float) * conv4_1_in_channel * conv4_1_out_channel *
                   conv4_1_kernel_size * conv4_1_kernel_size);
    cudaMalloc((void**)&d_conv4_1_bias, sizeof(float) * conv4_1_out_channel);
    cudaMalloc((void**)&d_conv4_2_weight,
               sizeof(float) * conv4_2_in_channel * conv4_2_out_channel *
                   conv4_2_kernel_size * conv4_2_kernel_size);
    cudaMalloc((void**)&d_conv4_2_bias, sizeof(float) * conv4_2_out_channel);
    cudaMalloc((void**)&d_conv4_3_weight,
               sizeof(float) * conv4_3_in_channel * conv4_3_out_channel *
                   conv4_3_kernel_size * conv4_3_kernel_size);
    cudaMalloc((void**)&d_conv4_3_bias, sizeof(float) * conv4_3_out_channel);

    //////////BLOCK 5/////////////////////////////////
    cudaMalloc((void**)&d_conv5_1_weight,
               sizeof(float) * conv5_1_in_channel * conv5_1_out_channel *
                   conv5_1_kernel_size * conv5_1_kernel_size);
    cudaMalloc((void**)&d_conv5_1_bias, sizeof(float) * conv5_1_out_channel);
    cudaMalloc((void**)&d_conv5_2_weight,
               sizeof(float) * conv5_2_in_channel * conv5_2_out_channel *
                   conv5_2_kernel_size * conv5_2_kernel_size);
    cudaMalloc((void**)&d_conv5_2_bias, sizeof(float) * conv5_2_out_channel);
    cudaMalloc((void**)&d_conv5_3_weight,
               sizeof(float) * conv5_3_in_channel * conv5_3_out_channel *
                   conv5_3_kernel_size * conv5_3_kernel_size);
    cudaMalloc((void**)&d_conv5_3_bias, sizeof(float) * conv5_3_out_channel);

    //////////FC 1////////////////////////////////////
    cudaMalloc((void**)&d_fc1_weight,
               sizeof(float) * fc1_in_channel * fc1_out_channel);
    cudaMalloc((void**)&d_fc1_bias, sizeof(float) * fc1_out_channel);

    // Alloc Activations
    cudaMalloc((void**)&d_image,
               sizeof(uint8_t) * batch * input_size * input_size * input_channel);
    cudaMalloc((void**)&d_input,
               sizeof(float) * batch * input_channel * input_size * input_size);

    //////////BLOCK 1/////////////////////////////////
    cudaMalloc((void**)&d_input_padded,
               sizeof(float) * batch * input_channel * (input_size + 2 * conv1_1_padding_size) * (input_size + 2 * conv1_1_padding_size));
    cudaMalloc((void**)&d_C1_1_feature_map,
               sizeof(float) * batch * C1_1_channel * C1_1_size * C1_1_size);
    cudaMalloc((void**)&d_C1_1_feature_map_padded,
               sizeof(float) * batch * C1_1_channel * (C1_1_size + 2 * conv1_2_padding_size) * (C1_1_size + 2 * conv1_2_padding_size));
    cudaMalloc((void**)&d_C1_2_feature_map,
               sizeof(float) * batch * C1_2_channel * C1_2_size * C1_2_size);
    cudaMalloc((void**)&d_S1_feature_map,
               sizeof(float) * batch * S1_channel * S1_size * S1_size);

    //////////BLOCK 2/////////////////////////////////
    cudaMalloc((void**)&d_S1_feature_map_padded,
               sizeof(float) * batch * S1_channel * (S1_size + 2 * conv2_1_padding_size) * (S1_size + 2 * conv2_1_padding_size));
    cudaMalloc((void**)&d_C2_1_feature_map,
               sizeof(float) * batch * C2_1_channel * C2_1_size * C2_1_size);
    cudaMalloc((void**)&d_C2_1_feature_map_padded,
               sizeof(float) * batch * C2_1_channel * (C2_1_size + 2 * conv2_2_padding_size) * (C2_1_size + 2 * conv2_2_padding_size));
    cudaMalloc((void**)&d_C2_2_feature_map,
               sizeof(float) * batch * C2_2_channel * C2_2_size * C2_2_size);
    cudaMalloc((void**)&d_S2_feature_map,
               sizeof(float) * batch * S2_channel * S2_size * S2_size);

    //////////BLOCK 3/////////////////////////////////
    cudaMalloc((void**)&d_S2_feature_map_padded,
               sizeof(float) * batch * S2_channel * (S2_size + 2 * conv3_1_padding_size) * (S2_size + 2 * conv3_1_padding_size));
    cudaMalloc((void**)&d_C3_1_feature_map,
               sizeof(float) * batch * C3_1_channel * C3_1_size * C3_1_size);
    cudaMalloc((void**)&d_C3_1_feature_map_padded,
               sizeof(float) * batch * C3_1_channel * (C3_1_size + 2 * conv3_2_padding_size) * (C3_1_size + 2 * conv3_2_padding_size));
    cudaMalloc((void**)&d_C3_2_feature_map,
               sizeof(float) * batch * C3_2_channel * C3_2_size * C3_2_size);
    cudaMalloc((void**)&d_C3_2_feature_map_padded,
               sizeof(float) * batch * C3_2_channel * (C3_2_size + 2 * conv3_3_padding_size) * (C3_2_size + 2 * conv3_3_padding_size));
    cudaMalloc((void**)&d_C3_3_feature_map,
               sizeof(float) * batch * C3_3_channel * C3_3_size * C3_3_size);
    cudaMalloc((void**)&d_S3_feature_map,
               sizeof(float) * batch * S3_channel * S3_size * S3_size);

    //////////BLOCK 4/////////////////////////////////
    cudaMalloc((void**)&d_S3_feature_map_padded,
               sizeof(float) * batch * S3_channel * (S3_size + 2 * conv4_1_padding_size) * (S3_size + 2 * conv4_1_padding_size));
    cudaMalloc((void**)&d_C4_1_feature_map,
               sizeof(float) * batch * C4_1_channel * C4_1_size * C4_1_size);
    cudaMalloc((void**)&d_C4_1_feature_map_padded,
               sizeof(float) * batch * C4_1_channel * (C4_1_size + 2 * conv4_2_padding_size) * (C4_1_size + 2 * conv4_2_padding_size));
    cudaMalloc((void**)&d_C4_2_feature_map,
               sizeof(float) * batch * C4_2_channel * C4_2_size * C4_2_size);
    cudaMalloc((void**)&d_C4_2_feature_map_padded,
               sizeof(float) * batch * C4_2_channel * (C4_2_size + 2 * conv4_3_padding_size) * (C4_2_size + 2 * conv4_3_padding_size));
    cudaMalloc((void**)&d_C4_3_feature_map,
               sizeof(float) * batch * C4_3_channel * C4_3_size * C4_3_size);
    cudaMalloc((void**)&d_S4_feature_map,
               sizeof(float) * batch * S4_channel * S4_size * S4_size);

    //////////BLOCK 5/////////////////////////////////
    cudaMalloc((void**)&d_S4_feature_map_padded,
               sizeof(float) * batch * S4_channel * (S4_size + 2 * conv5_1_padding_size) * (S4_size + 2 * conv5_1_padding_size));
    cudaMalloc((void**)&d_C5_1_feature_map,
               sizeof(float) * batch * C5_1_channel * C5_1_size * C5_1_size);
    cudaMalloc((void**)&d_C5_1_feature_map_padded,
               sizeof(float) * batch * C5_1_channel * (C5_1_size + 2 * conv5_2_padding_size) * (C5_1_size + 2 * conv5_2_padding_size));
    cudaMalloc((void**)&d_C5_2_feature_map,
               sizeof(float) * batch * C5_2_channel * C5_2_size * C5_2_size);
    cudaMalloc((void**)&d_C5_2_feature_map_padded,
               sizeof(float) * batch * C5_2_channel * (C5_2_size + 2 * conv5_3_padding_size) * (C5_2_size + 2 * conv5_3_padding_size));
    cudaMalloc((void**)&d_C5_3_feature_map,
               sizeof(float) * batch * C5_3_channel * C5_3_size * C5_3_size);
    cudaMalloc((void**)&d_S5_feature_map,
               sizeof(float) * batch * S5_channel * S5_size * S5_size);

    cudaMalloc((void**)&d_output, sizeof(float) * batch * output_size);

    // Copy Parameters
    //////////BLOCK 1/////////////////////////////////
    cudaMemcpy(d_conv1_1_weight, conv1_1_weight,
               sizeof(float) * conv1_1_in_channel * conv1_1_out_channel *
                   conv1_1_kernel_size * conv1_1_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_1_bias, conv1_1_bias, sizeof(float) * conv1_1_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_2_weight, conv1_2_weight,
               sizeof(float) * conv1_2_in_channel * conv1_2_out_channel *
                   conv1_2_kernel_size * conv1_2_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_2_bias, conv1_2_bias, sizeof(float) * conv1_2_out_channel,
               cudaMemcpyHostToDevice);

    //////////BLOCK 2/////////////////////////////////
    cudaMemcpy(d_conv2_1_weight, conv2_1_weight,
               sizeof(float) * conv2_1_in_channel * conv2_1_out_channel *
                   conv2_1_kernel_size * conv2_1_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_1_bias, conv2_1_bias, sizeof(float) * conv2_1_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_2_weight, conv2_2_weight,
               sizeof(float) * conv2_2_in_channel * conv2_2_out_channel *
                   conv2_2_kernel_size * conv2_2_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_2_bias, conv2_2_bias, sizeof(float) * conv2_2_out_channel,
               cudaMemcpyHostToDevice);

    //////////BLOCK 3/////////////////////////////////
    cudaMemcpy(d_conv3_1_weight, conv3_1_weight,
               sizeof(float) * conv3_1_in_channel * conv3_1_out_channel *
                   conv3_1_kernel_size * conv3_1_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_1_bias, conv3_1_bias, sizeof(float) * conv3_1_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_2_weight, conv3_2_weight,
               sizeof(float) * conv3_2_in_channel * conv3_2_out_channel *
                   conv3_2_kernel_size * conv3_2_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_2_bias, conv3_2_bias, sizeof(float) * conv3_2_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_3_weight, conv3_3_weight,
               sizeof(float) * conv3_3_in_channel * conv3_3_out_channel *
                   conv3_3_kernel_size * conv3_3_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_3_bias, conv3_3_bias, sizeof(float) * conv3_3_out_channel,
               cudaMemcpyHostToDevice);

    //////////BLOCK 4/////////////////////////////////
    cudaMemcpy(d_conv4_1_weight, conv4_1_weight,
               sizeof(float) * conv4_1_in_channel * conv4_1_out_channel *
                   conv4_1_kernel_size * conv4_1_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_1_bias, conv4_1_bias, sizeof(float) * conv4_1_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_2_weight, conv4_2_weight,
               sizeof(float) * conv4_2_in_channel * conv4_2_out_channel *
                   conv4_2_kernel_size * conv4_2_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_2_bias, conv4_2_bias, sizeof(float) * conv4_2_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_3_weight, conv4_3_weight,
               sizeof(float) * conv4_3_in_channel * conv4_3_out_channel *
                   conv4_3_kernel_size * conv4_3_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_3_bias, conv4_3_bias, sizeof(float) * conv4_3_out_channel,
               cudaMemcpyHostToDevice);

    //////////BLOCK 5/////////////////////////////////
    cudaMemcpy(d_conv5_1_weight, conv5_1_weight,
               sizeof(float) * conv5_1_in_channel * conv5_1_out_channel *
                   conv5_1_kernel_size * conv5_1_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_1_bias, conv5_1_bias, sizeof(float) * conv5_1_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_2_weight, conv5_2_weight,
               sizeof(float) * conv5_2_in_channel * conv5_2_out_channel *
                   conv5_2_kernel_size * conv5_2_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_2_bias, conv5_2_bias, sizeof(float) * conv5_2_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_3_weight, conv5_3_weight,
               sizeof(float) * conv5_3_in_channel * conv5_3_out_channel *
                   conv5_3_kernel_size * conv5_3_kernel_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_3_bias, conv5_3_bias, sizeof(float) * conv5_3_out_channel,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_fc1_weight, fc1_weight,
               sizeof(float) * fc1_in_channel * fc1_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_bias, fc1_bias, sizeof(float) * fc1_out_channel,
               cudaMemcpyHostToDevice);

    // copy input image
    size_t image_size = batch * input_size * input_size * input_channel;
    cudaMemcpy(d_image, image, image_size * sizeof(uint8_t),
               cudaMemcpyHostToDevice);

    // im2col
    cudaMalloc((void**)&d_input_padded_im2col,
               sizeof(float) * batch * input_channel * (C1_1_size + 2 * conv1_1_padding_size) * (C1_1_size + 2 * conv1_1_padding_size) * conv1_1_kernel_size * conv1_1_kernel_size);
    cudaMalloc((void**)&d_C1_1_feature_map_padded_im2col,
               sizeof(float) * batch * C1_1_channel * (C1_2_size + 2 * conv1_2_padding_size) * (C1_2_size + 2 * conv1_2_padding_size) * conv1_2_kernel_size * conv1_2_kernel_size);
    cudaMalloc((void**)&d_S1_feature_map_padded_im2col,
               sizeof(float) * batch * S1_channel * (C2_1_size + 2 * conv2_1_padding_size) * (C2_1_size + 2 * conv2_1_padding_size) * conv2_1_kernel_size * conv2_1_kernel_size);
    cudaMalloc((void**)&d_C2_1_feature_map_padded_im2col,
               sizeof(float) * batch * C2_1_channel * (C2_2_size + 2 * conv2_2_padding_size) * (C2_2_size + 2 * conv2_2_padding_size) * conv2_2_kernel_size * conv2_2_kernel_size);
    cudaMalloc((void**)&d_S2_feature_map_padded_im2col,
               sizeof(float) * batch * S2_channel * (C3_1_size + 2 * conv3_1_padding_size) * (C3_1_size + 2 * conv3_1_padding_size) * conv3_1_kernel_size * conv3_1_kernel_size);
    cudaMalloc((void**)&d_C3_1_feature_map_padded_im2col,
               sizeof(float) * batch * C3_1_channel * (C3_2_size + 2 * conv3_2_padding_size) * (C3_2_size + 2 * conv3_2_padding_size) * conv3_2_kernel_size * conv3_2_kernel_size);
    cudaMalloc((void**)&d_C3_2_feature_map_padded_im2col,
               sizeof(float) * batch * C3_2_channel * (C3_3_size + 2 * conv3_3_padding_size) * (C3_3_size + 2 * conv3_3_padding_size) * conv3_3_kernel_size * conv3_3_kernel_size);
    cudaMalloc((void**)&d_S3_feature_map_padded_im2col,
               sizeof(float) * batch * S3_channel * (C4_1_size + 2 * conv4_1_padding_size) * (C4_1_size + 2 * conv4_1_padding_size) * conv4_1_kernel_size * conv4_1_kernel_size);
    cudaMalloc((void**)&d_C4_1_feature_map_padded_im2col,
               sizeof(float) * batch * C4_1_channel * (C4_2_size + 2 * conv4_2_padding_size) * (C4_2_size + 2 * conv4_2_padding_size) * conv4_2_kernel_size * conv4_2_kernel_size);
    cudaMalloc((void**)&d_C4_2_feature_map_padded_im2col,
               sizeof(float) * batch * C4_2_channel * (C4_3_size + 2 * conv4_3_padding_size) * (C4_3_size + 2 * conv4_3_padding_size) * conv4_3_kernel_size * conv4_3_kernel_size);
    cudaMalloc((void**)&d_S4_feature_map_padded_im2col,
               sizeof(float) * batch * S4_channel * (C5_1_size + 2 * conv5_1_padding_size) * (C5_1_size + 2 * conv5_1_padding_size) * conv5_1_kernel_size * conv5_1_kernel_size);
    cudaMalloc((void**)&d_C5_1_feature_map_padded_im2col,
               sizeof(float) * batch * C5_1_channel * (C5_2_size + 2 * conv5_2_padding_size) * (C5_2_size + 2 * conv5_2_padding_size) * conv5_2_kernel_size * conv5_2_kernel_size);
    cudaMalloc((void**)&d_C5_2_feature_map_padded_im2col,
               sizeof(float) * batch * C5_2_channel * (C5_3_size + 2 * conv5_3_padding_size) * (C5_3_size + 2 * conv5_3_padding_size) * conv5_3_kernel_size * conv5_3_kernel_size);
}

void vgg16_cuda::classify(int* predict, int batch) {
    // read logits back to cpu
    cudaMemcpy(output, d_output, sizeof(float) * output_size * batch,
               cudaMemcpyDeviceToHost);
    // Softmax
    softmax(output, predict, batch, output_size);
}

vgg16_cuda::~vgg16_cuda() {
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

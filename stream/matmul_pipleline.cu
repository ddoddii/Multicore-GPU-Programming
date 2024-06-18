#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_WIDTH 16

// Define the matrix structure
struct Matrix
{
    float *elements;
    int width;
    int height;
};

// Matrix multiplication kernel using shared memory
__global__ void MatMulKernel(float *A, float *B, float *C, int width)
{
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0;

    // Loop over the A and B tiles required to compute the C element
    for (int t = 0; t < (width - 1) / TILE_WIDTH + 1; ++t)
    {
        // Load tiles into shared memory
        if (Row < width && t * TILE_WIDTH + tx < width)
            As[ty][tx] = A[Row * width + t * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0;

        if (t * TILE_WIDTH + ty < width && Col < width)
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * width + Col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        // Multiply tiles
        for (int k = 0; k < TILE_WIDTH; ++k)
            Cvalue += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    // Write the block result to the output matrix
    if (Row < width && Col < width)
        C[Row * width + Col] = Cvalue;
}

// Function to initialize matrices and execute the kernel
void ExecuteMatMul()
{
    // Assume matrix size
    int width = 1024; // Example width, can be changed
    size_t matSize = width * width * sizeof(float);

    // Allocate host matrices
    Matrix h_A, h_B, h_C;
    h_A.width = h_B.width = h_C.width = width;
    h_A.height = h_B.height = h_C.height = width;
    h_A.elements = new float[width * width];
    h_B.elements = new float[width * width];
    h_C.elements = new float[width * width];

    // Initialize matrices with some values
    for (int i = 0; i < width * width; ++i)
    {
        h_A.elements[i] = static_cast<float>(i % 100);
        h_B.elements[i] = static_cast<float>((i + 1) % 100);
        h_C.elements[i] = 0.0f;
    }

    // Allocate device matrices
    Matrix d_A, d_B, d_C;
    d_A.width = d_B.width = d_C.width = width;
    d_A.height = d_B.height = d_C.height = width;
    cudaMalloc(&d_A.elements, matSize);
    cudaMalloc(&d_B.elements, matSize);
    cudaMalloc(&d_C.elements, matSize);

    // Streams for asynchronous execution
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    // Copy host matrices to device
    cudaMemcpyAsync(d_A.elements, h_A.elements, matSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B.elements, h_B.elements, matSize, cudaMemcpyHostToDevice, stream2);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch matrix multiplication kernel
    MatMulKernel<<<dimGrid, dimBlock, 0, stream1>>>(d_A.elements, d_B.elements, d_C.elements, width);

    // Copy result matrix back to host
    cudaMemcpyAsync(h_C.elements, d_C.elements, matSize, cudaMemcpyDeviceToHost, stream1);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Cleanup: Destroy streams and free device memory
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    std::cout << "Result matrix C (portion): " << std::endl;
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            std::cout << h_C.elements[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free host memory
    delete[] h_A.elements;
    delete[] h_B.elements;
    delete[] h_C.elements;
}

int main()
{
    ExecuteMatMul();
    return 0;
}
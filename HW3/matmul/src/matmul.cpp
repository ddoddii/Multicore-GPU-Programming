#include "matmul.h"
#include <omp.h>
#include <pthread.h>
#include <vector>

#define THREADS 32

void matmul_ref(const int *const matrixA, const int *const matrixB, int *const matrixC, const int n, const int m)
{
    // You can assume matrixC is initialized with zero
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < m; k++)
                matrixC[i * n + j] += matrixA[i * m + k] * matrixB[k * n + j];
}

void transpose(const int *const matrixB, int *const matrixB_trans, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < m; k++)
        {
            matrixB_trans[k * n + i] = matrixB[i * m + k];
        }
    }
}

void matmul_optimized(const int *const matrixA, const int *const matrixB, int *const matrixC, const int n, const int m)
{
    std::vector<int> B_trans(m * n);
    transpose(matrixB, B_trans.data(), n, m);
#pragma omp parallel for shared(matrixA, matrixB, matrixC) schedule(static) num_threads(THREADS)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;
            for (int k = 0; k < m; k++)
            {
                sum += matrixA[i * m + k] * B_trans[j * m + k];
            }
            matrixC[i * n + j] = sum;
        }
    }
}
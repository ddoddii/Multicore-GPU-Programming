#include "matmul.h"
#include <omp.h>
#include <pthread.h>
#include <vector>

#define THREADS 32

int NT = THREADS;
int NT_T = THREADS;
int CASE = 0;
int B = 32;

void matmul_ref(const int *const matrixA, const int *const matrixB, int *const matrixC, const int n, const int m)
{
    // You can assume matrixC is initialized with zero
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < m; k++)
                matrixC[i * n + j] += matrixA[i * m + k] * matrixB[k * n + j];
}

void matmul_optimized(const int *const matrixA, const int *const matrixB, int *const matrixC, const int n, const int m)
{
    /*Transpose*/
    std::vector<int> matrixB_T(m * n);
    if (1)
    {
#pragma omp parallel for num_threads(NT_T) schedule(auto) collapse(2)
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
                matrixB_T[j * m + i] = matrixB[i * n + j];
        }
    }

    omp_set_num_threads(NT);

    /*Block Multiplication*/
    int local_sum = 0;
    switch (CASE)
    {
    case 0:
#pragma omp parallel for schedule(auto) collapse(2) firstprivate(local_sum)
        // B : Block size
        // Iterate over outer block
        for (int i = 0; i < n; i += B)
        {
            for (int j = 0; j < n; j += B)
            {
                for (int k = 0; k < m; k += B)
                {

                    // Iterate over elements within each block
                    for (int ii = 0; ii < B; ii++)
                    {
                        for (int jj = 0; jj < B; jj++)
                        {
                            for (int kk = 0; kk < B; kk++)
                            {
                                local_sum += matrixA[(i + ii) * m + (k + kk)] * matrixB[(j + jj) + (k + kk) * n];
                            }
                            matrixC[(i + ii) * n + (j + jj)] += local_sum;
                            local_sum = 0;
                        }
                    }
                }
            }
        }
        break;

    case 1:
        // For NB * NB submatrix - main
        local_sum = 0;
#pragma omp parallel for schedule(auto) collapse(2) firstprivate(local_sum)
        for (int i = 0; i < (n / B) * B; i += B)
        {
            for (int j = 0; j < (n / B) * B; j += B)
            {
                for (int k = 0; k < (m / B) * B; k += B)
                {

                    for (int ii = 0; ii < B; ii++)
                    {
                        for (int jj = 0; jj < B; jj++)
                        {
                            for (int kk = 0; kk < B; kk++)
                            {
                                local_sum += matrixA[(i + ii) * m + (k + kk)] * matrixB[(j + jj) + (k + kk) * n];
                            }
                            matrixC[(i + ii) * n + (j + jj)] += local_sum;
                            local_sum = 0;
                        }
                    }
                }
            }
        }

        // For NB * NB submatrix - remain
        local_sum = 0;
#pragma omp parallel for schedule(auto) collapse(2) firstprivate(local_sum)
        for (int i = 0; i < (n / B) * B; i += B)
        {
            for (int j = 0; j < (n / B) * B; j += B)
            {
                for (int k = (m / B) * B; k < m; k++)
                {
                    local_sum += matrixA[i * m + k] * matrixB[j + k * n];
                }
                matrixC[i * n + j] += local_sum;
                local_sum = 0;
            }
        }

        // For NB * r submatrix
        local_sum = 0;
#pragma omp parallel for schedule(auto) collapse(2) firstprivate(local_sum)
        for (int i = 0; i < (n / B) * B; i += B)
        {
            for (int j = (n / B) * B; j < n; j += B)
            {
                for (int k = 0; k < m; k++)
                {
                    local_sum += matrixA[i * m + k] * matrixB[j + k * n];
                }
                matrixC[i * n + j] += local_sum;
                local_sum = 0;
            }
        }

        // For r * NB submatrix
        local_sum = 0;
#pragma omp parallel for schedule(auto) collapse(2) firstprivate(local_sum)
        for (int i = (n / B) * B; i < n; i++)
        {
            for (int j = (n / B) * B; j < n; j++)
            {
                for (int k = 0; k < m; k++)
                {
                    local_sum += matrixA[i * m + k] * matrixB[j + k * n];
                }
                matrixC[i * n + j] += local_sum;
                local_sum = 0;
            }
        }
        break;

    default:
        break;
    }
}
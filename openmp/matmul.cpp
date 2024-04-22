#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

void matrixMultiply(const vector<vector<int>> &matrixA, const vector<vector<int>> &matrixB, vector<vector<int>> &result,
                    bool useOpenMP)
{
    int rowsA = matrixA.size();
    int colsA = matrixA[0].size();
    int colsB = matrixB[0].size();

    if (useOpenMP)
    {
#pragma omp parallel for
        for (int i = 0; i < rowsA; ++i)
        {
            for (int j = 0; j < colsB; ++j)
            {
                for (int k = 0; k < colsA; ++k)
                {
                    result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < rowsA; ++i)
        {
            for (int j = 0; j < colsB; ++j)
            {
                for (int k = 0; k < colsA; ++k)
                {
                    result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
    }
}

int main()
{
    int rowsA = 1000;
    int colsA = 1000;
    int colsB = 1000;

    vector<vector<int>> matrixA(rowsA, vector<int>(colsA, 1));
    vector<vector<int>> matrixB(colsA, vector<int>(colsB, 1));
    vector<vector<int>> result(rowsA, vector<int>(colsB, 0));

    // Matrix multiplication without OpenMP
    auto start = omp_get_wtime();
    matrixMultiply(matrixA, matrixB, result, false);
    auto end = omp_get_wtime();
    cout << "Matrix multiplication without OpenMP took " << end - start << " seconds." << endl;

    // Reset the result matrix
    result.assign(rowsA, vector<int>(colsB, 0));

    // Matrix multiplication with OpenMP
    start = omp_get_wtime();
    matrixMultiply(matrixA, matrixB, result, true);
    end = omp_get_wtime();
    cout << "Matrix multiplication with OpenMP took " << end - start << " seconds." << endl;

    return 0;
}
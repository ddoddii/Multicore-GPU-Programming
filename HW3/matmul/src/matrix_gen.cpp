#include <fstream>
#include <iostream>
#include <random>

// Function to generate a matrix and write it to the stream
void generateAndWriteMatrix(std::ofstream &file, std::vector<std::vector<int>> &matrix, int rows, int cols)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 10);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int value = distrib(gen);
            matrix[i][j] = value; // Store the value in the matrix
            file << value << (j == cols - 1 ? "\n" : " ");
        }
    }
}

void multiplyMatrices(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B,
                      std::vector<std::vector<int>> &C, int n, int m)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C[i][j] = 0; // Initialize element to zero
            for (int k = 0; k < m; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void writeMatrixToFile(const std::vector<std::vector<int>> &matrix, std::ofstream &file)
{
    for (const auto &row : matrix)
    {
        for (int val : row)
        {
            file << val << " ";
        }
        file << "\n";
    }
}

int main()
{
    int n = 2048;
    int m = 2048;
    std::vector<std::vector<int>> A(n, std::vector<int>(m));
    std::vector<std::vector<int>> B(m, std::vector<int>(n));
    std::vector<std::vector<int>> C(n, std::vector<int>(n));

    std::ofstream matrixFileIn("matrix2048.in");

    // Write dimensions
    matrixFileIn << n << " " << m << "\n";

    // Generate and write matrix A (n x m)
    generateAndWriteMatrix(matrixFileIn, A, n, m);

    // Generate and write matrix B (m x n)
    generateAndWriteMatrix(matrixFileIn, B, m, n);

    // Close the input file
    matrixFileIn.close();

    // Perform matrix multiplication
    multiplyMatrices(A, B, C, n, m);

    // Open the output file for matrix C
    std::ofstream matrixFileOut("matrix2048.out");

    // Write the resulting matrix C to file
    writeMatrixToFile(C, matrixFileOut);

    // Close the output file
    matrixFileOut.close();

    return 0;
}

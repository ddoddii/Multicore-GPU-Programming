#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

/*
Matrix multiplication using single thread / multiple threads
=== sample run ===
1) N = 1000, num_threads = 4
single thread : 2986 milliseconds
multiple threads : 775 milliseconds
*/

int **A, **B, **C;
int N;

void multiply_row(int start_row, int end_row)
{
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void initialize_matrices()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = dis(gen);
            B[i][j] = dis(gen);
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix size> <num threads>\n";
        return 1;
    }

    N = std::stoi(argv[1]);
    int num_threads = std::stoi(argv[2]);

    A = new int *[N];
    B = new int *[N];
    C = new int *[N];
    for (int i = 0; i < N; i++)
    {
        A[i] = new int[N];
        B[i] = new int[N];
        C[i] = new int[N];
    }

    initialize_matrices();

    // Single thread calculation
    auto start = std::chrono::high_resolution_clock::now();
    multiply_row(0, N);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken by single thread: " << duration.count() << " milliseconds\n";

    // Multi-thread calculation
    std::vector<std::thread> threads;
    int rows_per_thread = N / num_threads;
    start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < num_threads; t++)
    {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? N : start_row + rows_per_thread;
        threads.emplace_back(multiply_row, start_row, end_row);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken by multiple threads: " << duration.count() << " milliseconds\n";

    // Clean up
    for (int i = 0; i < N; i++)
    {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

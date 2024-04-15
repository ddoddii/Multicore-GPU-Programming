#include <assert.h>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

std::atomic<int> output;

void worker(int *input, int start, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (input[start + i] == 0)
        {
            output += 1;
        }
    }
}

int main()
{
    const int arraySize = 1000;
    const int numThreads = 4;

    int input[arraySize];

    for (int i = 0; i < arraySize; i++)
    {
        input[i] = (i % 2 == 0) ? 0 : 1;
    }

    std::vector<std::thread> threads;
    int chunkSize = arraySize / numThreads;

    // Spawn multiple threads to execute the worker function
    for (int i = 0; i < numThreads; i++)
    {
        int start = i * chunkSize;
        int size = (i == numThreads - 1) ? (arraySize - start) : chunkSize;
        threads.emplace_back(worker, input, start, size);
    }

    // Wait for all threads to finish
    for (auto &thread : threads)
    {
        thread.join();
    }

    // Check the final value of output
    int expectedOutput = arraySize / 2; // Half of the array elements are 0
    assert(output == expectedOutput);

    std::cout << "Test passed. Final output: " << output << std::endl;

    return 0;
}
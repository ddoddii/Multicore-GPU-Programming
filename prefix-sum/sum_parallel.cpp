#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

void worker(int *input, int start, int size, int *output)
{
    for (int i = start; i < start + size; ++i)
    {
        *output += input[i];
    }
}

int main()
{
    const int arraySize = 100000000;
    int *input = new int[arraySize];
    for (int i = 0; i < arraySize; ++i)
    {
        input[i] = i;
    }
    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::vector<int> partialSums(numThreads, 0);

    int chunkSize = arraySize / numThreads;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numThreads; ++i)
    {
        int start = i * chunkSize;
        threads.push_back(std::thread(worker, input, start, chunkSize, &partialSums[i]));
    }

    for (auto &t : threads)
    {
        t.join();
    }

    int totalSum = 0;
    for (const auto &sum : partialSums)
    {
        totalSum += sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Total Sum: " << totalSum << std::endl;
    std::cout << "Time taken (parallel): " << duration.count() << " seconds" << std::endl;

    delete[] input;
    return 0;
}
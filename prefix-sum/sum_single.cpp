#include <chrono>
#include <iostream>

void worker(int *input, int size, int *output)
{
    for (int i = 0; i < size; ++i)
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
    int totalSum = 0;

    auto start = std::chrono::high_resolution_clock::now();

    worker(input, arraySize, &totalSum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Total Sum: " << totalSum << std::endl;
    std::cout << "Time taken (non-parallel): " << duration.count() << " seconds" << std::endl;

    delete[] input;
    return 0;
}
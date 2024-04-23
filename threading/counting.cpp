#include <iostream>
#include <thread>
#include <vector>

/*
Race condition because of multi-threads using the same resource
*/

void worker(int *input, int start, int size, int *output)
{
    for (int i = 0; i < size; i++)
    {
        if (input[start + i] == 0)
        {
            (*output)++;
        }
    }
}

int main(int argc, char *argv[])
{
    const int N = atoi(argv[1]);
    const int NT = atoi(argv[2]);

    int *array = new int[N];
    for (int i = 0; i < N; i++)
    {
        array[i] = 0;
    }

    int count = 0;
    std::vector<std::thread> threads;
    for (int t = 0; t < NT; t++)
    {
        int size = N / NT;
        int start = t * size;
        threads.push_back(std::thread(worker, array, start, size, &count));
    }
    for (auto &thread : threads)
    {
        thread.join();
    }

    std::cout << "There are " << count << " zeros" << std::endl;
}
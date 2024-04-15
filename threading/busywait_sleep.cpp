#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

std::mutex m;

std::queue<int> shared_queue;
const int N = 10000;
void produce()
{
    for (int i = 0; i < N; i++)
    {
        m.lock();
        std::cout << "i produce " << i << std::endl;
        shared_queue.push(i);
        m.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void sleep_consume()
{
    for (int i = 0; i < N; i++)
    {
        while (shared_queue.empty())
        {
            m.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            m.lock();
        }
        std::cout << "i read " << shared_queue.front() << std::endl;
        shared_queue.pop();
    }
}

int main()
{
    std::thread t1(produce);
    std::thread t2(sleep_consume);
    t1.join();
    t2.join();
    return 0;
}
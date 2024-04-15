#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

std::mutex m;

std::queue<int> shared_queue;
const int N = 10000;
std::condition_variable cond; // condition variable

void produce()
{
    for (int i = 0; i < N; i++)
    {
        std::unique_lock<std::mutex> lock(m);
        std::cout << "i produce " << i << std::endl;
        shared_queue.push(i);
        cond.notify_one();
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void consume()
{
    for (int i = 0; i < N; i++)
    {
        std::unique_lock<std::mutex> lock(m);
        cond.wait(lock, [] { return !shared_queue.empty(); });
        std::cout << "i read " << shared_queue.front() << std::endl;
        shared_queue.pop();
        lock.unlock();
    }
}

int main()
{
    std::thread t1(produce);
    std::thread t2(consume);
    t1.join();
    t2.join();
    return 0;
}
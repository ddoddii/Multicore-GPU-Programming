#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

class Barrier
{
  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int count_;
    int initial_count_;

  public:
    explicit Barrier(int count) : count_(count), initial_count_(count)
    {
    }

    void Wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (--count_ == 0)
        {
            count_ = initial_count_;
            cv_.notify_all();
        }
        else
        {
            cv_.wait(lock, [this] { return count_ == initial_count_; });
        }
    }
};

void workerThread(Barrier &barrier)
{
    // Phase 1
    std::cout << "Thread " << std::this_thread::get_id() << " doing phase 1 work\n";

    // Wait for all threads to complete phase 1
    barrier.Wait();

    // Phase 2
    std::cout << "Thread " << std::this_thread::get_id() << " doing phase 2 work\n";
}

int main()
{
    const int numThreads = 4;
    Barrier barrier(numThreads);

    std::thread threads[numThreads];

    // Create and start the threads
    for (int i = 0; i < numThreads; ++i)
    {
        threads[i] = std::thread(workerThread, std::ref(barrier));
    }

    // Wait for all threads to finish
    for (int i = 0; i < numThreads; ++i)
    {
        threads[i].join();
    }

    return 0;
}
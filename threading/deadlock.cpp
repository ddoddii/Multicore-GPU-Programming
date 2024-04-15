#include <iostream>
#include <thread>
#include <vector>

std::mutex global_mutex;
class int_wrapper

{
  public:
    int_wrapper(int val) : val(val)
    {
    }
    std::mutex m;
    int val;
};

void swap(int_wrapper &v1, int_wrapper &v2)
{
    std::lock(v1.m, v2.m);
    std::lock_guard<std::mutex> lock_v1(v1.m, std::adopt_lock);
    std::lock_guard<std::mutex> lock_v2(v2.m, std::adopt_lock);
    int tmp = v1.val;
    v1.val = v2.val;
    v2.val = tmp;
}

int main()
{
    int_wrapper a(0);
    int_wrapper x(1);

    for (int i = 0; i < 10000; i++)
    {
        std::cout << "start iteration " << i << std::endl;
        std::thread t1(swap, std::ref(x), std::ref(a));
        std::thread t2(swap, std::ref(a), std::ref(x));

        t1.join();
        t2.join();
        std::cout << "done a : " << a.val << ", x:" << x.val << std::endl;
    }
    return 0;
}
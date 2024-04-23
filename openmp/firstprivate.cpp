#include <cassert>
#include <cstdio>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    omp_set_num_threads(2);
    int is_private = -2;
/*
firstprivate : initialized to value outside region
Modifying is_private within parallel block does not modify the value outside the block
*/
#pragma omp parallel firstprivate(is_private)
    {
        int tid = omp_get_thread_num();
        printf("Thread ID %2d  | is_private(before) = %d\n", tid, is_private);
        is_private = tid;
        printf("Thread ID %2d  | is_private(after) = %d\n", tid, is_private);
        assert(is_private == tid);
    }
    printf("Main thread  | is_private = %d\n", is_private);

    return 0;
}

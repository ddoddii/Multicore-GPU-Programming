#include <cassert>
#include <cstdio>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    omp_set_num_threads(2);
    int last_private = -2;
/*
lastprivate : not initialized,
becomes what's written in the last iteration (no matter when, which thread executed it!)
*/
#pragma omp parallel for lastprivate(last_private)
    for (int i = 0; i < 10; i++)
    {
        int tid = omp_get_thread_num();
        printf("Thread ID %2d excuting i=%d  | last_private(before) = %d\n", tid, i, last_private);
        last_private = i;
        printf("Thread ID %2d excuting i=%d  | last_private(after) = %d\n", tid, i, last_private);
        assert(last_private == i);
    }
    printf("Main thread  | last_private = %d\n", last_private);

    return 0;
}

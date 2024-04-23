#include <cassert>
#include <cstdio>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    omp_set_num_threads(10);

#pragma omp parallel
    {
/*
master + explicit barrier works like "single"
*/
#pragma omp master
        for (int i = 0; i < 5; i++)
        {
            int tid = omp_get_thread_num();
            printf("Thread ID %2d section A\n", tid);
        }
#pragma omp barrier
#pragma omp fpr
        for (int i = 0; i < 5; i++)
        {
            int tid = omp_get_thread_num();
            printf("Thread ID %2d section B\n", tid);
        }
    }

    return 0;
}

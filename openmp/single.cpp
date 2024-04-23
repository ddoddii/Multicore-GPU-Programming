#include <cassert>
#include <cstdio>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    omp_set_num_threads(10);

#pragma omp parallel
    {
#pragma omp single
        for (int i = 0; i < 10; i++)
        {
            int tid = omp_get_thread_num();
            printf("Thread ID %2d section A\n", tid);
        } // implicit barrier !!

#pragma omp for
        for (int i = 0; i < 100; i++)
        {
            int tid = omp_get_thread_num();
            printf("Thread ID %2d section B\n", tid);
        }
    }

    return 0;
}

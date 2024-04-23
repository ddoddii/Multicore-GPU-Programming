#include <iostream>
#include <omp.h>

int main(int argc, char **argv)
{
    omp_set_num_threads(10);
    int sum = 0;

#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < 100; i++)
    {
        sum += i;
    }
    printf("Sum : %d\n", sum);
    return 0;
}
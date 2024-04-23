#include <cstdio>
#include <omp.h>

int main()
{
    long long int c[100], k[100], a[100], b[100];
    // Dummy initialization for demonstration
    for (int i = 0; i < 100; ++i)
    {
        a[i] = i;
        b[i] = 100 - i;
        k[i] = i % 10;
        c[i] = 0;
    }

// Using static scheduling
#pragma omp parallel for schedule(static)
    for (long long int i = 0; i < 100; i++)
    {
        c[i] += k[i] * a[i];
        c[i] += k[i] * b[i];
    }

    // Using dynamic scheduling with a chunk size of 10
    /*
    #pragma omp parallel for schedule(dynamic, 10)
    for (long long int i = 0; i < 100; i++) {
        c[i] += k[i] * a[i];
        c[i] += k[i] * b[i];
    }
    */

    // Using guided scheduling with a minimum chunk size of 5
    /*
    #pragma omp parallel for schedule(guided, 5)
    for (long long int i = 0; i < 100; i++) {
        c[i] += k[i] * a[i];
        c[i] += k[i] * b[i];
    }
    */

    // Using auto scheduling
    /*
    #pragma omp parallel for schedule(auto)
    for (long long int i = 0; i < 100; i++) {
        c[i] += k[i] * a[i];
        c[i] += k[i] * b[i];
    }
    */

    // Optionally print results to verify correctness
    for (int i = 0; i < 10; ++i)
    { // Print first 10 results for brevity
        printf("c[%d] = %lld\n", i, c[i]);
    }

    return 0;
}

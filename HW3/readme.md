# Matrix Multiplication

## Background

Given two matrix A[n * m], B[m * n], let's optimize **matrix multiplication** A*B. The serial version of this is below.

```cpp
void matmul(const int *const matrixA, const int *const matrixB, int *const matrixC, const int n, const int m)
{
    // You can assume matrixC is initialized with zero
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < m; k++)
                matrixC[i * n + j] += matrixA[i * m + k] * matrixB[k * n + j];
}
```

## Optimizing Matmul

There are many ways to optimize matrix multiplication.

### 1. Transpose matrix

In matrix multiplication A*B, A is read row-wise but B is read column-wise. Reading a matrix column-wise decreases cache hit rate, because when you read an element in a matrix, the row is read together. Because of cache size, cache miss willl occur because of capacity miss.

Therefore, we can transpose matrix B, thereby making it row-wise read.

```cpp
void transpose(const int *const matrixB, int *const matrixB_trans, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < m; k++)
        {
            matrixB_trans[k * n + i] = matrixB[i * m + k];
        }
    }
}
```

### 2. OpenMP

**OpenMP** is a set of compiler directives as well as an API for programs written in C, C++, or FORTRAN that provides support for parallel programming in shared-memory environments. By using openmp, programmers can give information to compilers that the code written needs to be processed in parallel. Openmp automatically makes appropriate number of threads for parallel programming.


-  `parallel` : Code block (`{}`) that comes after parallel is executed in multi threads.
-  `shared(matrixA, matrixB, matrixC)` : Make variables matrixA, matrixB, matrixC shared among all threads.
- `schedule(dynamic,-1)` : Divide the tasks into chunck size, and allocate remaining task to the threads that finished executing.

```cpp
void matmul_optimized(const int *const matrixA, const int *const matrixB, int *const matrixC, const int n, const int m)
{
    std::vector<int> B_trans(m * n);
    transpose(matrixB, B_trans.data(), n, m);
#pragma omp parallel for shared(matrixA, matrixB, matrixC) schedule(dynamic) num_threads(THREADS)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;
            for (int k = 0; k < m; k++)
            {
                sum += matrixA[i * m + k] * B_trans[j * m + k];
            }
            matrixC[i * n + j] = sum;
        }
    }
}
```


### 3. Matrix Blocks

We can also split matrix into blocks to fit the block into L1 cache. It's similar to using divide-and-conquer techinque. I split the matrix into 64*64 blocks, but the matmul was slower than not dividing into blocks. 


<img width="491" alt="image" src="https://github.com/ddoddii/Multicore-GPU-Programming/assets/95014836/eb44000a-06bd-4a4d-9e5e-488d87edc88a">


## Result

The best result was when I used transpose + openmp.

<img width="499" alt="image" src="https://github.com/ddoddii/Multicore-GPU-Programming/assets/95014836/cdd5d719-5176-4afc-97d0-ab88b450f44f">




## Reference
- https://learn.microsoft.com/ko-kr/cpp/parallel/openmp/reference/openmp-clauses?view=msvc-170

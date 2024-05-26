# Matrix Multiplication

## 1. Introduction

### 1.1 Objective

Given two matrix A[n * m], B[m * n], the objective is to optimize **matrix multiplication** A*B. The size of matrix is restricted to be within the range of 256 <= n,m <= 2048.


### 1.2 Hardware Spec

The given server uses **AMD EPYC 7452 32-Core Processor**. 

![image](https://github.com/ddoddii/Multicore-GPU-Programming/assets/95014836/f627935e-32aa-41a3-b759-6abfa32af1e1)

Figure 1. Hardware Configuration using `lscpu`


| AMD EPYC 7452 32-Core Processor |                              |
|---------------------------------|------------------------------|
| **Core(s) per socket**          | 32                           |
| **L1i cache**                   | 32 x 32KiB 8-way set associative |
| **L1d cache**                   | 32 x 32KiB 8-way set associative Write-Back |
| **L2 cache**                    | 32 x 512KiB 8-way set associative Write-Back |
| **L3 cache**                    | 8 x 16MiB                    |
| **Memory block size**           | 128M                         |
| **Total online memory**         | 512G                         |

Table 1. Hardware Configuration

![image](https://github.com/ddoddii/Multicore-GPU-Programming/assets/95014836/b9673b88-544b-4723-8ec2-b885718497f5)

Figure 2. EPYC 7452 - Cache Hierarchy

To optimize specially for our hardware environment, let’s briefly take a look at hardware configuration. CPU model name is **AMD EPYC 7452 32-Core Processor**. We can see that there is 32 physical cores per socket and 1MB L1 I cache, D cache, 32 MB L2 cache, 512 MB L3 cache and 512GB of main memory. L1 is dedicated for each core as a local cache whose size is 32KB each with 8-way set associativity. L2 cache is also dedicated for each core and its size is 512KB each. L3 cache is shared among the cores, which is dedicated for each CPU Complex(CCX). Figure 2. represents the cache hierarchy of each CPU Complex(CCX). CPU Complex(CCX) is basic building blocks of AMD CPU, which consists of several physical cores and shared resources. Each CCX of EPYC 7542 consists of 4 physical cores, L2 cache and L3 cache. Later on, we’ll optimize our algorithm to maximize the utilization of these cache composition.


## 2. Background

### 2.1 Memory Access

As mentioned at assignment1, [HW1: 1D Parallel Filtering], one of the most important thing in parallel computing is memory access. Since most of parallel computing application is actually memory bounded, it is important to optimize the memory access. Let’s start with how memory access actually occurs in real system.

<img width="493" alt="image" src="https://github.com/ddoddii/Multicore-GPU-Programming/assets/95014836/8c4a3087-3ebe-41ae-a17a-65eaac36c66a">

Figure 3. Memory Hierarchy

In the Von Neumann architecture, there are two essential components: the **processor** and the **memory**. While the speed of processor gets faster rapidly, speed of memory remains relatively slow. This was a conventional bottleneck of Von Neumann architecture. To address this bottleneck, the concept of a **cache** was introduced. By positioning a fast, small cache close to the processor and leveraging the locality of data access, it e↵ectively reduces the memory access latency. Cache closer to the processor is faster but small and in this perspec- tive register can be treated as L0 cache. Conversely, further cache is bigger but slower and main memory can be treated as the last level cache.

When processor requests memory I/O, it first accesses to the highest level of cache(e.g. L1 cache). If there is data that matches to the request, it performs the request(read/write). This is called cache hit. On the other hand, if there is no data that matches to the request, it accesses the lower level of cache and so on. This is called cache miss. Since higher level of cache is faster, if cache hit occurs for most of the memory access, system can reduce the memory access latency effectively, and might be amortized the overhead of cache hierarchy which potentially leads to performance enhancement.

<img width="215" alt="image" src="https://github.com/ddoddii/Multicore-GPU-Programming/assets/95014836/01cbe4c7-9c41-41e2-9714-13c035f0da43">

Figure 4. Data Transfer between processor - cache - main memory

Figure 4. shows how data is actually transferred between processor, cache and main memory. When data is transferred between different level of cache (or main memory), data adjacent to the requested one is also transferred. This unit of memory accessing is called **cache block** or **cache line**. Therefore, it might be note worthy to **align the data along to cache block size** when distributing the workload to multiple threads can actually improve the performance. This alignment might improve the cache utilization and also lower the false sharing problem.

Dealing with the memory access latency, **cache miss** must be primarily considered. There is 3 type of cache miss: **Cold miss, Capacity miss and Conflict miss**. **Cold miss**, also called compulsory miss refers to the cache miss which occurs at the every first access of certain cache block which is inevitable. **Capacity miss** refers to the cache miss due to the size of cache. If data is larger than the cache size, processor needs to access beyond the cache size which occurs eviction of cache block. Therefore, it is very important to adjust the data to fit the cache size. **Conflict miss** refers to the cache miss due to the overlapped cache block address. It can be understood exactly the same to conflict in hash table. Since it leads to cache under utilization where eviction occurs even there is sufficient space, it must be mitigated by reorganizing the data or changing the memory access pattern.

Now, let’s look at memory access in matrix multiplication. Assume cache block size is C and multiplying N x N square matrix A and B and each matrix is allocated to memory in row-major order. Matrix A will be accessed in row-major, therefore we can fully utilize the both spatial and temporal locality of data. $\frac{N}{C}$ cache miss will occur for a single row,
and there is N rows, $\frac{N^2}{C}$ cache miss will occur. On the other hand, matrix B is accessed C in column-major, N miss occur for computing a single element of matrix C. Therefore, $\frac{N^3}{C}$ cache miss occurs for matrix B. Also, in the case matrix B doesn’t fit into the cache, it might suffer from excessive cache miss due to eviction.

### 2.2 Transpose

Accessing in **column major order** is problematic. Especially, when column size N is multiple of 2, due to conflict miss, cache utilization lower with factor of its associativity. Consider situation where cache blocks of same column is mapped to same set. Assume the cache is A-way set associative, first A access will bring the cache block without eviction. However after that, eviction occurs for every C access and it leads to conflict miss where C represents the size of cache block.

To mitigate this, techniques such as padding or transpose can be done. Padding focuses on changing the data organization. By padding, we can adjust the cache blocks from the same column do not mapped to same set. On the other hand, transpose actually focuses on changing the memory access pattern. By transposing the matrix, processor accesses BT in row major order which mitigates the excessive cache miss due to conflicts. Also it helps improving the cache utilization in the case matrix size is bigger than cache size.

### 2.3 Blocked Matrix Multiplication

**Blocked matrix multiplication**, also known as **tiling** is some kind of divide-and-conquer technique. When matrix size is larger than cache size, miss frequency increases due to eviction(conflict miss). To achieve high performance on large matrices, blocked matrix mul- tiplication techniques divdes entire matrix into small sub-matrices and perform matrix mul- tiplication on them.

Since it performs computation repeatedly on sub-matrix, it can fully utilize the **temporal locality** of the data. Direct approach occurs $\frac{N^2}{C}$  misses on A and $\frac{N^3}{C}$ misses on B, where C represents for the size of cache block. Using blocked matrix multiplication, it reduces to $\frac{N^3}{B \times C}$ both for matrix A and B where B represents for size of block. It’s note worthy that cache miss is inversely proportional to the size of block B. Therefore, we can induce that maximizing the block B while maintaining each block fit into the L1 cache is desired.

Blocked matrix multiplication is particularly e↵ective when a single row doesn’t fit to L1 cache. If a single row fit into the L1 cache, matrix A occurs $\frac{N^2}{C}$ misses. On the other hand, if it doesn’t fit into L1 cache, matrix A occurs $\frac{N^3}{C}$ misses due to eviction. Therefore, by dividing the entire matrix into small, well-fit sub-matrices we can maintain the $\frac{N^3}{B \times C}$ read misses even for the large size of matrix by utilizing the temporal locality.

However, we also must be mindful of the overhead of this technique. First of all, it leads to increased level of for loop. Second, additional computation for remaining elements might be required in the case of matrix size doesn’t divided perfectly with block size. We’ll analyze about this later.

Also, considering about the layout of sub-matrix can be a good point. We’ll start with $B \times B$ sub-matrix first, and apply other layout such as $B_1 \times B_2$  sub-matrix to optimize depending on the input matrices.

### 2.4 OpenMP

**OpenMP** is a set of compiler directives as well as an API for programs written in C, C++, or FORTRAN that provides support for parallel programming in shared-memory environments. By using openmp, programmers can give information to compilers that the code written needs to be processed in parallel. Openmp automatically makes appropriate number of threads for parallel programming.


-  `parallel` : Code block (`{}`) that comes after parallel is executed in multi threads.
-  `shared(matrixA, matrixB, matrixC)` : Make variables matrixA, matrixB, matrixC shared among all threads.
- `schedule(dynamic,-1)` : Divide the tasks into chunck size, and allocate remaining task to the threads that finished executing.

## 3. Implementation

### 3.1 Direct Approach 

```cpp
omp_set_num_threads(NT);
#pragma omp parallel for firstprivate(local_sum)
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++){
        for (int k = 0; k < m; k++)
            local_sum += matrixA[i * m + k] * matrixB[k * n + j];
    matrixC[i*n+j] += local_sum; 
    local_sum = 0;
    } 
}
```

Above code represents the most simple way to implement matrix multiplication. It is just the same with the given reference matrix multiplication code except two points. First, **parallel for clause** in OpenMP directive is used to parallelize for loop. Also, private variable **`local_sum`** is used to minimize the memory access. When each thread has its own `local_sum`, the variable is likely stored in the thread's local CPU cache (L1 cache). Accessing the L1 cache is much faster compared to accessing shared memory. This reduces the latency of accessing and updating the variable. To make local sum private for each thread, firstprivate clause is used.

### 3.2 Transpose

```cpp
/*Transpose*/
if(1){
    #pragma omp parallel for num_threads(NT_T) schedule(auto) collapse(2) 
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++) 
            matrixB_T[j*m+i] = matrixB[i*n+j];
    } 
}

omp_set_num_threads(NT);
/*Matrix Multiplication*/
#pragma omp prallel for collapse(2) schedule(auto) firstprivate(local_sum)
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++){
        for (int k = 0; k < m; k++)
            local_sum += matrixA[i * m + k] * matrixB_T[j * m + k];
    matrixC[i * n + j] += local_sum; 
    local_sum = 0;
    } 
}
```

In matrix multiplication A*B, A is read row-wise but B is read column-wise. Reading a matrix column-wise decreases cache hit rate, because when you read an element in a matrix, the row is read together. Because of cache size, cache miss willl occur because of capacity miss.

Above code represents matrix multiplication with **transposing matrix B**. Both transposition and matrix multiplication are executed in parallel using OpenMP directive.

### 3.3 Block Matrix Multiplication

```cpp
switch(CASE)
{
    case 0:
    #pragma omp parallel for schedule(auto) collapse(2) firstprivate( local_sum)
    // B : Block size 
    // Iterate over outer block
    for(int i=0; i<n; i+=B){ 
        for(int j=0; j<n; j+=B){
            for(int k=0; k<m; k+=B){

                // Iterate over elements within each block
                for(int ii=0; ii<B; ii++){ 
                    for(int jj=0; jj<B; jj++){
                        for(int kk=0; kk<B; kk++){
                            local_sum += matrixA[(i+ii)*m + (k+kk)] * matrixB[(j+jj) + (k+kk)*n]; 
                            }
                            matrixC[(i+ii)*n + (j+jj)] += local_sum;
                            local_sum =0; 
                        }
                } 
            }         
        } 
    }
    break;

    /* 
    Edge case : If n is not multiple of B 
    n = k * B + r
    (n/B) * B ~= k * B (Largest Integer smaller than n that is multiple of B)
    */
    case 1:
    // For NB * NB submatrix - main
    #pragma omp parallel for schedule(auto) collapse(2) firstprivate( local_sum)
    for(int i=0; i<(n/B)*B; i+=B){ 
        for(int j=0; j<(n/B)*B; j+=B){
            for(int k=0; k<(m/B)*B; k+=B){

                for(int ii=0; ii<B; ii++){ 
                    for(int jj=0; jj<B; jj++){
                        for(int kk=0; kk<B; kk++){
                            local_sum += matrixA[(i+ii)*m + (k+kk)] * matrixB[(j+jj) + (k+kk)*n]; 
                            }
                            matrixC[(i+ii)*n + (j+jj)] += local_sum;
                            local_sum =0; 
                    }
                } 
            }         
        } 
    }

    // For NB * NB submatrix - remain 
    local_sum = 0;
    #pragma omp parallel for schedule(auto) collapse(2) firstprivate(
local_sum)
    for(int i=0; i<(n/B)*B; i+=B){ 
        for(int j=0; j<(n/B)*B; j+=B){
            for(int k=(m/B)*B; k<m; k++){
                local_sum += matrixA[i*m+k] * matrixB[j+k*n];
            }
            matrixC[i*n+j] += local_sum;
            local_sum = 0;         
        } 
    }

    // For NB * r submatrix
    local_sum = 0;
    #pragma omp parallel for schedule(auto) collapse(2) firstprivate( local_sum)
    for(int i=0; i<(n/B)*B; i+=B){ 
        for(int j=(n/B)*B; j<n; j+=B){
            for(int k=0; k<m; k++){
                local_sum += matrixA[i*m+k] * matrixB[j+k*n];
            }
            matrixC[i*n+j] += local_sum;
            local_sum = 0;         
        } 
    }

    // For r * NB submatrix
    local_sum = 0;
    #pragma omp parallel for schedule(auto) collapse(2) firstprivate( local_sum)
    for(int i=(n/B)*B; i<n; i++){ 
        for(int j=(n/B)*B; j<n; j++){
            for(int k=0; k<m; k++){
                local_sum += matrixA[i*m+k] * matrixB[j+k*n];
            }
            matrixC[i*n+j] += local_sum;
            local_sum = 0;         
        } 
    }
    break;

    default:
    break;
}
```

<img width="213" alt="image" src="https://github.com/ddoddii/Multicore-GPU-Programming/assets/95014836/a118ae40-0bf6-4d78-8529-d782a0c94be5">

Figure 5. Blocked Matrix Multiplication


Above code shows implementation of **blocked matrix multiplication**(or tiling). There exists two cases, one for the case where **input matrices divided into bloc**k and the **other case where they don’t**. Latter case needs additional computation for remaining elements. First for loop computes on NB x NB sub-matrix of matrix C using blocked matrix multiplication tech- niques. Second for loop also computes on NB x NB sub-matrix of matrix C, but for calculate the elements that doesn’t included in prior computation. Third for loop computes on NB x R1 sub-matrix of matrix C. Fourth loop computes on R2 x NB sub-matrix of matrix C. Last loop computes on R1 x R2 sub-matrix of matrix C. As we can see, in the case where input matrices doesn’t divided by block, additional computation required to get the correct results. Constructing additional loop might be a significant overhead especially for matrices with small size. Therefore, we need to consider the e↵ect of size and appropriately handle this.



## 4. Evaluation & Optimzation

### 4.1 Evaluation

| Technique        | Matrix Size | Run Time [sec] |
|------------------|-------------|----------------|
| Direct Approach  | 1024, 1024  | 0.527784       |
|                  | 2048, 2048  | -              |
| Transpose        | 1024, 1024  | 0.107273       |
|                  | 2048, 2048  | 2.49538        |
| Tiling           | 1024, 1024  | 0.138311       |
|                  | 2048, 2048  | 1.85971        |
| All Together     | 1024, 1024  | 0.0389178      |
|                  | 2048, 2048  | 0.20426        |

Table 2. Evaluation on GEMM

I’ve done evaluation for each matrix multiplication introduced at [2. Implementation] part. I set 16 threads in the case of size=1024, and 32 threads in the case of size=2048. For Tiling, I set size of blocked matrix B=32. We can observe that applying all together, we can achieve more than 12 times speed up comparing to direct approach. The parameter set in this evaluation is done arbitrarily. It might be determined theoretically and experimentally on [4.2 Optimization] part.

### 4.2 Optimazion

#### 4.2.1 Block Size

Let's start with optimizing the size of sub-matrix. As mentioned earlier, since large B leads to more speed up, we need to choose maximum B while maintaining two sub-matrices of matrix A and B fit into L1 cache. L1 D cache for each core is 32KB, therefore we can simply calculate the optimum \( B = \sqrt{\frac{2^{15}}{2^2 \times 2}} = 2^6 \).

| Block Size (B) | Matrix Size | Run Time [sec] |
|----------------|-------------|----------------|
| B=32           | 1024, 1024  | 0.0389178      |
|                | 2048, 2048  | 0.20426        |
| B=64           | 1024, 1024  | 0.0330113      |
|                | 2048, 2048  | 0.0919427      |
| B=128          | 1024, 1024  | 0.0330148      |
|                | 2048, 2048  | 0.237377       |

Table 3: Evaluation on GEMM w.r.t Block Size **B**

Above table represents the runtime of GEMM with respecting to block size B. Therefore we can induce that the optimum block size B=64.

#### 4.2.2 Number of Threads - NT

The next parameter is **number of threads**. As mentioned at assignment 1, large number of threads doesn’t always lead to performance improvement. Since creating and distributing the workload for multiple threads have significant overheads especially workload or parallelism isn’t sufficient to amortize it, we should adjust the number of thread properly depends on the workloads.

| # of Threads (NT) | Matrix Size | Run Time [sec] |
|-------------------|-------------|----------------|
| NT=8              | 1024, 1024  | 0.0469192      |
|                   | 2048, 2048  | 0.196808       |
| NT=16             | 1024, 1024  | 0.0331326      |
|                   | 2048, 2048  | 0.129705       |
| NT=32             | 1024, 1024  | 0.0389415      |
|                   | 2048, 2048  | 0.0852724      |

Table 4: Evaluation on GEMM w.r.t NT

We can observe that for 1024 x 1024 matrices, **NT=16** is appropriate and for 2048 x 2048 matrices, NT = 32 is appropriate. Since size of input matrices can vary, I conducted additional experiment with various input matrices. In conclusion, I got the following results.

| Matrix Size | NT=4    | NT=8    | NT=16   | NT=24   | NT=32   | NT=36    |
|-------------|---------|---------|---------|---------|---------|----------|
| 256, 256    | 2.52885 | 2.86874 | 2.55238 | 3.10523 | 3.29498 | 4.448818 |
| 512, 512    | 13.1073 | 9.40423 | 7.51635 | 7.80597 | 8.00512 | 7.83803  |
| 768, 768    | 24.0846 | 23.6799 | 17.1232 | 14.6426 | 18.22   | 17.0669  |
| 1024, 1024  | 84.4475 | 39.9654 | 33.5514 | 32.2738 | 29.2203 | 27.748   |
| 1536, 1536  | 126.876 | 75.5891 | 57.7635 | 57.98   | 52.6268 | 46.1095  |
| 2048, 2048  | 289.211 | 160.722 | 114.011 | 101.759 | 104.664 | 77.3954  |

Table 5: Evaluation on GEMM w.r.t NT

Based on this results, I determined the NT depending on input matrices as follows.

| Matrix Size | NT |
|-------------|----|
| [256, 512)  | 8  |
| [512, 1024) | 24 |
| [1024, 2048)| 36 |

Table 6: NT depending on input matrices

#### 4.2.3 Number of Threads - NT_T

Now let’s determine the threads for transposing the matrix. Fixing the number of threads determined above, changing the number of threads used for transposition NT_T, I evaluated the performance. Based on this result, I determined NT_T as follows.

| Matrix Size | NT=4    | NT=8    | NT=16   | NT=24   | NT=32   | NT=36    |
|-------------|---------|---------|---------|---------|---------|----------|
| 256         | 1.67871 | 2.17986 | 2.85071 | 2.91372 | 15.2056 | 3.20369  |
| 512         | 6.36122 | 5.27901 | 7.23843 | 5.39373 | 11.9085 | 10.1852  |
| 768         | 15.7138 | 16.4575 | 15.7371 | 23.5842 | 26.704  | 18.556   |
| 1024        | 31.802  | 32.6474 | 28.9627 | 16.4664 | 42.9884 | 19.8939  |
| 1536        | 44.3026 | 47.7124 | 49.4893 | 52.2393 | 66.262  | 40.8807  |
| 2048        | 78.6685 | 140.888 | 72.4241 | 53.5816 | 234.383 | 88.5893  |

Table 7: Evaluation on GEMM w.r.t NT_T

| Matrix Size | [256, 512) | [512, 1024) | [1024, 2048) |
|-------------|------------|-------------|--------------|
| NT_T        | 4          | 14          | 24           |

Table 8: NT_T depending on input matrices




## Reference
[1] Cache Hierarchy: https://www.anandtech.com/show/14694/amd-rome-epyc-2nd-gen/7 
[2] HW configuration: https://en.wikichip.org/wiki/amd/epyc/7542
[3] BLIS: Smith, John, and Alice Doe. 2023. ”Anatomy of High-Performance Many- Threaded Matrix Multiplication.” Journal of Computational Mathematics 29 (2): 101-120


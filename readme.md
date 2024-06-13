# Multicore GPU Programming

The repository covers a wide range of topics, each aimed at improving efficiency and performance in **GPU programming**. Here’s a detailed look at what I learned: <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Nerd%20Face.png" alt="Nerd Face" width="25" height="25" />


## Theory 

|Theme|Post|
|--|--|
|Basic Parallel Architectures|[Basic Parallel Architectures에 대해 알아보자](https://ddoddii.github.io/post/cs/mgp/basic-parallel-architecture/)|
|Thread Programming|[c++로 알아본 쓰레드 프로그래밍](https://ddoddii.github.io/post/cs/mgp/threading/)|
|Thread Management|[멀티쓰레드에서 쓰레드 간 작업을 어떻게 균일하게 분할할까?](https://ddoddii.github.io/post/cs/mgp/thread-management/)|
|Matrix Multiplication (multi-threaded)|[멀티쓰레드에서 행렬 연산(matmul) 성능 증가시키는 방법들](https://ddoddii.github.io/post/cs/mgp/multithread-matmul/)|
|OpenMP|[멀티쓰레딩을 편리하게 해주는 OpenMP 사용법](https://ddoddii.github.io/post/cs/mgp/openmp/)|
|Graph Processing|[그래프 구조를 더 효율적으로 저장하는 방법들](https://ddoddii.github.io/post/cs/mgp/graph-processing/)|
|Prefix sum|[Prefix Sum : 효율적인 연산을 위한 가이드](https://ddoddii.github.io/post/cs/mgp/prefix-sum/#kogge-stone-algorithm)|
|CUDA Programming Intro|[CUDA 프로그래밍 기초](https://ddoddii.github.io/post/cs/mgp/cuda-programming/)|
|CPU-GPU communication and thread indexing|[CPU-GPU 통신 및 CUDA를 활용한 이미지 프로세싱 기법](https://ddoddii.github.io/post/cs/mgp/cuda-programming-2/)|
|CUDA thread hierarchy, memory hierarchy, GPU cache structure|[CUDA와 Nvidia GPU 아키텍처: 스레드 계층, 메모리 계층 및 GPU 캐시 구조 이해하기](https://ddoddii.github.io/post/cs/mgp/cuda-programming-3/)|

## Hands on Assignment

|Assignment|Description|Link|
|----|---|--|
|Assignment #1|A Simple Filter on 1D Array|[link](https://github.com/ddoddii/Multicore-GPU-Programming/tree/master/HW1)|
|Assignment #2|Hash table locking|[link](https://github.com/ddoddii/Multicore-GPU-Programming/tree/master/HW2)|
|Assignment #3|Matrix Multiplication|[link](https://github.com/ddoddii/Multicore-GPU-Programming/tree/master/HW3)|
|Assignment #4|Matrix Multiplication using CUDA|[link](https://github.com/ddoddii/Multicore-GPU-Programming/tree/master/HW4)|
|Assignment #5|Sum Reduction|[link](https://github.com/ddoddii/Multicore-GPU-Programming/tree/master/HW5)|
|Assignment #6|CUDA Application of DNN|[link](https://github.com/ddoddii/Multicore-GPU-Programming/tree/master/HW6)|


## Content Breakdown

### Basic Parallel Architectures

- Post: [Basic Parallel Architectures에 대해 알아보자](https://ddoddii.github.io/post/cs/mgp/basic-parallel-architecture/)
- Description: This section introduces the fundamental concepts of parallel architectures, laying the groundwork for more advanced topics.

### Thread Programming

- Post: [c++로 알아본 쓰레드 프로그래밍](https://ddoddii.github.io/post/cs/mgp/threading/)
- Description: Dive into thread programming with C++, understanding how to create and manage threads effectively.

### Thread Management

- Post: [멀티쓰레드에서 쓰레드 간 작업을 어떻게 균일하게 분할할까?](https://ddoddii.github.io/post/cs/mgp/thread-management/)
- Description: Learn strategies for evenly distributing tasks among threads in a multithreaded environment to maximize performance.

### Matrix Multiplication (multi-threaded)

- Post: [멀티쓰레드에서 행렬 연산(matmul) 성능 증가시키는 방법들](https://ddoddii.github.io/post/cs/mgp/multithread-matmul/)
- Description: Explore methods to optimize matrix multiplication operations using multithreading techniques.

### OpenMP

- Post: [멀티쓰레딩을 편리하게 해주는 OpenMP 사용법](https://ddoddii.github.io/post/cs/mgp/openmp/)
- Description: Get acquainted with OpenMP, a powerful tool that simplifies multithreading and parallel programming.

### Graph Processing

- Post: [그래프 구조를 더 효율적으로 저장하는 방법들](https://ddoddii.github.io/post/cs/mgp/graph-processing/)
- Description: Discover efficient ways to store and process graph structures, crucial for handling complex data relationships.

### Prefix Sum

- Post: [Prefix Sum : 효율적인 연산을 위한 가이드](https://ddoddii.github.io/post/cs/mgp/prefix-sum/#kogge-stone-algorithm)
- Description: Gain a comprehensive understanding of the prefix sum algorithm and its applications in efficient computation.

### CUDA 101
  
- Post : [CUDA 프로그래밍 기초](https://ddoddii.github.io/post/cs/mgp/cuda-programming/)
- Description : This section provides an introduction to CUDA programming, designed for those new to GPU programming. This post includes the basics of CUDA, including how to set up your development environment, write and compile your first CUDA program.

### CPU-GPU communication and thread indexing

- Post : [CPU-GPU 통신 및 CUDA를 활용한 이미지 프로세싱 기법](https://ddoddii.github.io/post/cs/mgp/cuda-programming-2/)
- Description : This section provides detailed explanation about the hierarchical structure of CUDA threads, including grids, blocks, and threads. This post includes calculating global thread index through thread indexing and some example code about image processing.

### CUDA thread hierarchy, memory hierarchy, GPU cache structure

- Post : [CUDA와 Nvidia GPU 아키텍처: 스레드 계층, 메모리 계층 및 GPU 캐시 구조 이해하기](https://ddoddii.github.io/post/cs/mgp/cuda-programming-3/)
- Description : This section delves into the advanced aspects of CUDA and Nvidia GPU architecture, including the hierarchical organization of threads, the different levels of memory, and the structure of GPU caches.
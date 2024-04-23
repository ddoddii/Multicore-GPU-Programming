# Hash Tables

## 1. Background

### Hash Table

Hash table uses a **hash function** to compute an *index*, also called *hash code*, into an array of buckets or slots. The index functions as a storage location for the matching value.

### Hash function

**Hash function** turns arbitrary string to fixed length. When you put in the same input, same output is generated.

Selecting a hash function is crucial, since uing a function that evenly distributes the keys and reduce hash collisions.

### Load factor

Load Factor is defined as “The number of entries in the table (n)” Divided by “The Size of the Hash Table (k)”

$$load \ factor = \frac{n}{k}$$

### Collision resolution techinques

Collisions happen when two or more keys point to the same array index. Chaining, open addressing, and double hashing are a few techniques for resolving collisions.

- **Open addressing** : When collision happens, just look for the next empty address. There are many ways to implement this, including double hashing, linear probing, quadratic probing.
- **Seperate Chaining** : This uses a linked list of objects that hash to each slot in the hash table. 

### Why is hash table fast?

For hash table, search, insert and delete are all O(1) on average and worst case O(n). O(1) means their execution time does not depend on the number of elements in the table. The hash value works as an index, so no matter how many elements are in the hash table, I don't have to search through the whole table (which takes O(n)), instead I can just use the index to find the position in the hash table.
 

## 2. Thread-safe Hash Table

To make a hash table thread-safe in multi threading environment, there are many possible options.

### sol1. Global lock

Global lock locks the entire hash table, meaning only one thread can access to the hash table. This is easy, but very slow.

<img width="533" alt="image" src="https://github.com/ddoddii/Multicore-GPU-Programming/assets/95014836/593b465f-81d7-483e-8f8f-c4e03200785b">


### sol2. Fine-grained lock

<img width="474" alt="image" src="https://github.com/ddoddii/Multicore-GPU-Programming/assets/95014836/02b38808-be20-4c53-ba9f-00e6ef5d6660">

Another option is using locks per bucket. This is called fine-grained locks. This is faster than global lock, but there is too much space overhead (O(B) : B is number of buckets)

### sol3. Lock striping

<img width="509" alt="image" src="https://github.com/ddoddii/Multicore-GPU-Programming/assets/95014836/e3c5ab94-ff2d-4adc-90ca-0d5abcf414c2">

Instead of using locks per bucket, we can use lock striping. Make an array of locks(size $L$), Lock $l$ manages item $i$ with $i%L=l$. There is a trade-off between performance and space.

## Reference
- https://en.wikipedia.org/wiki/Hash_table
- https://www.geeksforgeeks.org/hash-table-data-structure/
- https://www.reddit.com/r/cpp_questions/comments/3sxdal/why_are_hash_tables_faster_than_arrays/
- https://medium.com/nerd-for-tech/the-magic-of-hash-tables-a-quick-deep-dive-into-o-1-1295199fcd05
#ifndef _BETTER_LOCKED_HASH_TABLE_H_
#define _BETTER_LOCKED_HASH_TABLE_H_

#include "bucket.h"
#include "hash_table.h"
#include <iostream>
#include <mutex>
#include <thread>

class better_locked_probing_hash_table : public hash_table
{

  private:
    Bucket *table;
    const int TABLE_SIZE; // we do not consider resizing. Thus the table has to be larger than the max num items.
    // mutex global_mutex;

    /* TODO: put your own code here  (if you need something)*/
    /****************/
    std::mutex *segment_locks;
    /****************/
    /* TODO: put your own code here */

  public:
    better_locked_probing_hash_table(int table_size) : TABLE_SIZE(table_size)
    {
        this->table = new Bucket[TABLE_SIZE]();
        int num_locks = TABLE_SIZE / 5000;
        segment_locks = new std::mutex[num_locks];

        for (int i = 0; i < TABLE_SIZE; i++)
        {
            this->table[i].valid = 0; // means empty
        }
    }

    virtual uint32_t hash(uint32_t x)
    {
        // https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return (x % TABLE_SIZE);
    }

    virtual uint32_t hash_next(uint32_t key, uint32_t prev_index)
    {
        // linear probing. no special secondary hashfunction
        return ((prev_index + 1) % TABLE_SIZE);
    }

    // the buffer has to be allocated by the caller
    bool read(uint32_t key, uint64_t *value_buffer)
    {
        /* TODO: put your own read function here */
        /****************/
        uint64_t index = this->hash(key);
        int num_locks = TABLE_SIZE / 5000;
        uint32_t segment = index / (TABLE_SIZE / num_locks);
        std::lock_guard<std::mutex> lock(segment_locks[segment]);

        int probe_count = 0;

        while (table[index].valid == true)
        {
            if (table[index].key == key)
            {
                *value_buffer = table[index].value;
                return true;
            }
            else
            {
                probe_count++;
                index = this->hash_next(key, index);
                if (probe_count >= TABLE_SIZE)
                    break;
            }
        }
        return false;

        /****************/
        /* TODO: put your own read function here */
    }

    bool insert(uint32_t key, uint64_t value)
    {
        /* TODO: put your own insert function here */
        /****************/
        int num_locks = TABLE_SIZE / 5000;
        uint64_t index = this->hash(key);
        uint32_t segment = index / (TABLE_SIZE / num_locks);
        std::lock_guard<std::mutex> lock(segment_locks[segment]);
        int probe_count = 0;

        while (table[index].valid == true)
        {
            if (table[index].key == key)
            {
                // found it already there. just modify
                break;
            }
            else
            {
                probe_count++;
                index = this->hash_next(key, index);
                if (probe_count >= TABLE_SIZE)
                    return false; // could not add because the table was full
            }
        } // end while

        table[index].valid = true;
        table[index].key = key;
        table[index].value = value;
        return true;
        /****************/
        /* TODO: put your own insert function here */
    }

    int num_items()
    {
        int count = 0;
        for (int i = 0; i < TABLE_SIZE; i++)
        {
            if (table[i].valid == true)
                count++;
        }
        return count;
    }
};

#endif
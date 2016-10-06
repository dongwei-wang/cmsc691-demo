#include <stdio.h>
#include <stdint.h>
#include <stdatomic.h>
#include "c11threads.h"
 
atomic_int atomic_counter = 0;

int thread_atomic_counter(void* param)
{
    for(int i = 0; i < 1e6; i++) {
        atomic_counter++;
    }
}

int thread_regular_counter(void* param)
{
    int* local_counter = (int*) param;
    
    for(int i = 0; i < 1e6; i++) {
        *local_counter++;
    }
}
 
int main(void)
{
    struct timespec start, end;
    
    thrd_t thr[10];
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    for(int i = 0; i < 10; i++)
        thrd_create(&thr[i], thread_atomic_counter, NULL);
        
    for(int i = 0; i < 10; i++)
        thrd_join(thr[i], NULL);
        
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        
    printf("The atomic counter is %u\n", atomic_counter);
  
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
    printf("Elapsed atomic CPU time = %llu ms\n", (long long unsigned int) diff);
    
    
    
    
    // Local counters
    
    int counters[10];
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    for(int i = 0; i < 10; i++)
        counters[i] = 0;
    
    for(int i = 0; i < 10; i++)
        thrd_create(&thr[i], thread_regular_counter, &counters[i]);

    for(int i = 0; i < 10; i++)
        thrd_join(thr[i], NULL);

    int counter_sum = 0;
    
    for(int i = 0; i < 10; i++)
        counter_sum += counters[i]; // Reduce results
        
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
            
    printf("The sum of multiple local counters is %u\n", atomic_counter);
  
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
    printf("Elapsed regular CPU time = %llu ms\n", (long long unsigned int) diff);        
}

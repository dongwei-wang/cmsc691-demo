#include <stdio.h>
#include <stdatomic.h>
#include "c11threads.h"
 
atomic_int atomic_counter = 0;
int regular_counter = 0;
 
int thread(void* param)
{
    for(int i = 0; i < 1e6; i++) {
        atomic_counter++;
        regular_counter++;
    }
}
 
int main(void)
{
    thrd_t thr[10];
    
    for(int i = 0; i < 10; i++)
        thrd_create(&thr[i], thread, NULL);
        
    for(int i = 0; i < 10; i++)
        thrd_join(thr[i], NULL);
 
    printf("The atomic counter is %u\n", atomic_counter);
    printf("The non-atomic counter is %u\n", regular_counter);
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

void *thread() 
{
  // Initialize thread, load data, prepare computation
  
  // Do some cool stuff here
  
  pthread_exit(0);
}
 
int main(int argc, char *argv[])
{
  int i, j, n_threads = 4;
  pthread_t *threads;
  struct timespec start, end;
  
  threads = (pthread_t*)malloc(n_threads * sizeof(pthread_t));
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  for(j = 0; j < 100; j++)
  {
    for(i = 0; i < n_threads; i++)
      pthread_create(&threads[i],NULL,thread,NULL);
      
    for(i = 0; i < n_threads; i++)
      pthread_join(threads[i],NULL);    
      
    // Do some sequential stuff  
  }
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  
  uint64_t diff = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  
  printf("Elapsed process CPU time = %llu nanoseconds\n", (long long unsigned int) diff);
  
  free(threads);
}

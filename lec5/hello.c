#include <stdio.h>

void main()
{
    omp_set_num_threads(5);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("Hello world from thread %d, total threads %d\n", tid, nthreads);
    }
}

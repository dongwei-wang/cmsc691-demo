#include <stdio.h>

void main()
{
    int counter = 0;

    #pragma omp parallel num_threads(1000)
    {
        #pragma omp critical
        counter++;
    }

    printf("Counter critical %d\n", counter);

    counter = 0;

    #pragma omp parallel num_threads(1000)
    {
        #pragma omp atomic
        counter++;
    }

    printf("Counter atomic %d\n", counter);
}

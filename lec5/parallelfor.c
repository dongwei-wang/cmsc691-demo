#include <stdio.h>

void main()
{
    int size = 100;
    int *a, *b, *c;

    a = (int*)malloc(size * sizeof(int));
    b = (int*)malloc(size * sizeof(int));
    c = (int*)malloc(size * sizeof(int));

    #pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
        printf("Index %d thread %d total threads %d\n", i, omp_get_thread_num(), omp_get_num_threads());
    }

    free(a);
    free(b);
    free(c);
}

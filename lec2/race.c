#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

void* run1(void * arg)
{
    int * x = (int*) arg;
    *x += 1;
}

void *run2 (void * arg)
{
    int * x = (int *) arg;
    *x += 2;
}

int main(int argc, char * argv[])
{
    int * x = malloc (sizeof(int));
    *x = 1;
    pthread_t t1, t2;
    pthread_create(&t1, NULL, &run1, x);
    pthread_create(&t2, NULL, &run2, x);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf ("%d\n", *x);
    free(x);
}

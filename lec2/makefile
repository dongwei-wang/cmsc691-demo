todo: race vector_race vector_race_mutex producer_consumer sequential_parallel sequential_parallel_persistent_threads atomic atomic_performance
race: race.c
	gcc-4.9 -pthread -g race.c -o race
vector_race: vector_race.cpp
	g++-4.9 -std=c++11 -pthread -g vector_race.cpp -o vector_race
vector_race_mutex: vector_race_mutex.cpp
	g++-4.9 -std=c++11 -pthread -g vector_race_mutex.cpp -o vector_race_mutex
producer_consumer: producer_consumer.c
	gcc-4.9 -pthread -g producer_consumer.c -o producer_consumer
sequential_parallel: sequential_parallel.c
	gcc-4.9 -pthread -g sequential_parallel.c -o sequential_parallel
sequential_parallel_persistent_threads: sequential_parallel_persistent_threads.c
	gcc-4.9 -pthread -g sequential_parallel_persistent_threads.c -o sequential_parallel_persistent_threads
atomic: atomic.c
	gcc-4.9 -std=gnu11 -pthread -g atomic.c -o atomic
atomic_performance: atomic_performance.c
	gcc-4.9 -std=gnu11 -pthread -g atomic_performance.c -o atomic_performance
clean:
	rm race vector_race vector_race_mutex producer_consumer sequential_parallel sequential_parallel_persistent_threads atomic atomic_performance

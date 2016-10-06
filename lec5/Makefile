todo: hello parallelfor private reduction critical
hello: hello.c
	gcc-4.9 -fopenmp hello.c -o hello
parallelfor: parallelfor.c
	gcc-4.9 -std=gnu11 -fopenmp parallelfor.c -o parallelfor
private: private.c
	gcc-4.9 -std=gnu11 -fopenmp private.c -o private
reduction: reduction.c
	gcc-4.9 -std=gnu11 -fopenmp reduction.c -o reduction
critical: critical.c
	gcc-4.9 -std=gnu11 -fopenmp critical.c -o critical
clean:
	rm hello parallelfor private reduction critical

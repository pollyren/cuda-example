.PHONY: all clean

all: add_cpu add_gpu

add_cpu:
	gcc add.c -o add_cpu

add_gpu:
	nvcc add.cu -o add_gpu

clean:
	rm -f add_cpu add_gpu a.out
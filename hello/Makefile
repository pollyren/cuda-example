.PHONY: all clean

all: hello_cpu hello_gpu hello32_gpu

hello_cpu:
	gcc hello.c -o hello_cpu

hello_gpu:
	nvcc hello.cu -o hello_gpu

hello32_gpu:
	nvcc hello32.cu -o hello32_gpu

clean:
	rm -f hello_cpu hello_gpu hello32_gpu a.out
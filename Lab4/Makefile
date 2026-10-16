NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_20
PYTHON = python3

.PHONY: all example experiments clean

all: matrix_cuda

matrix_cuda: main.cu
	$(NVCC) $(NVCCFLAGS) main.cu -o matrix_cuda

example: matrix_cuda
	./matrix_cuda data/A.txt data/B.txt data/C.txt 16
	$(PYTHON) ../Lab1/verify.py data/A.txt data/B.txt data/C.txt

experiments: matrix_cuda
	$(PYTHON) cuda_experiments.py --binary ./matrix_cuda --sizes 200 400 800 1200 1600 2000 --tiles 4 8 16 32 --repeats 5 --output-dir results

clean:
	rm -f matrix_cuda matrix_cuda.exe data/C.txt

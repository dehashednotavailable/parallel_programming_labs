CXX = clang++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -Wpedantic
OMP_CXXFLAGS ?= -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
OMP_LDFLAGS ?= -L/opt/homebrew/opt/libomp/lib -lomp
PYTHON = python3

.PHONY: all example check experiments clean

all: lab2

lab2: main.cpp
	$(CXX) $(CXXFLAGS) $(OMP_CXXFLAGS) main.cpp $(OMP_LDFLAGS) -o lab2

example: lab2
	./lab2 data/A.txt data/B.txt data/C.txt 4
	$(PYTHON) ../Lab1/verify.py data/A.txt data/B.txt data/C.txt

check: lab2
	$(PYTHON) check.py --binary ./lab2

experiments: lab2
	$(PYTHON) lab2_experiments.py --binary ./lab2 --threads 1 2 4 8 10 --sizes 200 400 800 1200 1600 2000 --repeats 5 --output-dir results

clean:
	rm -f lab2 lab2.exe data/C.txt

CXX = mpicxx
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -Wpedantic
PYTHON = python3

.PHONY: all example check experiments clean

all: lab3

lab3: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o lab3

example: lab3
	mpirun --oversubscribe -np 4 ./lab3 data/A.txt data/B.txt data/C.txt
	$(PYTHON) ../Lab1/verify.py data/A.txt data/B.txt data/C.txt

check: lab3
	$(PYTHON) check.py --binary ./lab3

experiments: lab3
	$(PYTHON) mpi_experiments.py --binary ./lab3 --processes 1 2 4 8 --sizes 200 400 800 1200 1600 2000 --repeats 5 --output-dir results --launcher 'mpirun --oversubscribe'

clean:
	rm -f lab3 lab3.exe data/C.txt

CXX = c++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -Wpedantic
PYTHON = python3

.PHONY: all example check experiments clean

all: lab1

lab1: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o lab1

example: lab1
	./lab1 data/A.txt data/B.txt data/C.txt
	$(PYTHON) verify.py data/A.txt data/B.txt data/C.txt

check: lab1
	$(PYTHON) check.py --binary ./lab1

experiments: lab1
	$(PYTHON) run_experiments.py --binary ./lab1 --sizes 100 200 400 800 1200 1600 2000 --repeats 5 --compiler "$(CXX)" --flags "$(CXXFLAGS)" --output-dir results

clean:
	rm -f lab1 lab1.exe

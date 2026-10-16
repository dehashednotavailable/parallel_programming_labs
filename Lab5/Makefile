CXX = c++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -Wpedantic -pthread
PYTHON = python3

.PHONY: all check experiments clean

all: mandelbrot

mandelbrot: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o mandelbrot

check: mandelbrot
	$(PYTHON) check.py --binary ./mandelbrot

experiments: mandelbrot
	$(PYTHON) run_benchmarks.py --binary ./mandelbrot --repeats 3 --output-dir results

clean:
	rm -f mandelbrot mandelbrot.exe

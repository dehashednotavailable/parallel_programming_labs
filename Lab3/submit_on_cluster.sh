#!/bin/bash

source /etc/profile
set -euo pipefail
cd "$(dirname "$0")"

module load intel/mpi5
make clean
make CXX=mpicxx CXXFLAGS='-O2 -std=c++11 -Wall -Wextra -Wpedantic'
mkdir -p results/sergey-korolev
sbatch run_on_cluster.sbatch "$@"

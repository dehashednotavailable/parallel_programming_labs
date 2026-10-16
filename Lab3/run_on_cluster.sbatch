#!/bin/bash
#SBATCH --job-name=lab3-mpi
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=02:00:00
#SBATCH --output=results/sergey-korolev/slurm-%j.out

source /etc/profile
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load intel/mpi5

python3 ../tools/cluster_matrix_experiments.py \
    --program mpi \
    --binary ./lab3 \
    --configurations 1 2 4 8 \
    --sizes 200 400 800 1200 1600 2000 \
    --repeats 5 \
    --seed 42 \
    --output-dir results/sergey-korolev \
    --launcher srun \
    --mpi-version 'Intel MPI 5.1.3' \
    --compiler mpicxx \
    --flags '-O2 -std=c++11 -Wall -Wextra -Wpedantic'

#!/bin/bash
#SBATCH --job-name=lab4-cuda
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/sergey-korolev/slurm-%j.out

source /etc/profile
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load cuda/8.0
make clean
make NVCCFLAGS='-O2 -arch=sm_20'

bash ../tools/cluster_cuda_experiments.sh ./matrix_cuda results/sergey-korolev 5

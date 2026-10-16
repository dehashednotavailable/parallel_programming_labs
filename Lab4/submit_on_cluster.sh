#!/bin/bash

source /etc/profile
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p results/sergey-korolev
sbatch run_on_cluster.sbatch "$@"

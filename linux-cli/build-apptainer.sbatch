#!/bin/bash
#============ Slurm Options ===========
#SBATCH -p gr10561g
#SBATCH -t 00:10:00
#SBATCH --rsc p=1:t=32:c=56:m=16G
#SBATCH -J toy-acai-build
#SBATCH -o logs/%x.%j.out
#============ Shell Script ============

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
srun ./build-apptainer.sh

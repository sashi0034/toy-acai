#!/bin/bash
#============ Slurm Options ===========
#SBATCH -p gr10561g
#SBATCH -t 00:10:00
#SBATCH --rsc p=1:t=32:c=56:m=16G
#SBATCH -J toy-acai-build
#SBATCH -o logs/%x.%j.out
#============ Shell Script ============

set -euo pipefail

REPO_ROOT=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)

cd "${REPO_ROOT}/linux-cli"
srun ./build-apptainer.sh

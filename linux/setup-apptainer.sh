#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

mkdir -p ~/container_image

apptainer build --fakeroot \
  ~/container_image/toy-acai-ubuntu22.sif \
  "${SCRIPT_DIR}/toy-acai.def"

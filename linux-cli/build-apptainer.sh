#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=${TOY_ACAI_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
REPO_ROOT=${TOY_ACAI_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}

SIV3D_ROOT=${SIV3D_ROOT:-"${HOME}/ws/siv3d/siv6"}
SIV3D_APPTAINER_IMAGE=${SIV3D_APPTAINER_IMAGE:-"${HOME}/container_image/toy-acai-ubuntu22.sif"}
BUILD_DIR=${BUILD_DIR:-"${SCRIPT_DIR}/build"}
BUILD_PARALLELISM=${BUILD_PARALLELISM:-1}

if [[ ! -f "${SIV3D_APPTAINER_IMAGE}" ]]; then
	echo "Apptainer image was not found: ${SIV3D_APPTAINER_IMAGE}" >&2
	exit 1
fi

if [[ ! -f "${SIV3D_ROOT}/Linux/build/libSiv3D.a" ]]; then
	echo "libSiv3D.a was not found: ${SIV3D_ROOT}/Linux/build/libSiv3D.a" >&2
	exit 1
fi

apptainer exec \
	--bind "${REPO_ROOT}:${REPO_ROOT}" \
	--bind "${SIV3D_ROOT}:${SIV3D_ROOT}" \
	"${SIV3D_APPTAINER_IMAGE}" \
	env \
		TEST_ROOT="${SCRIPT_DIR}" \
		SIV3D_ROOT="${SIV3D_ROOT}" \
		BUILD_DIR="${BUILD_DIR}" \
		BUILD_PARALLELISM="${BUILD_PARALLELISM}" \
	bash -lc '
		set -euo pipefail

		cmake -GNinja \
			-S "${TEST_ROOT}" \
			-B "${BUILD_DIR}" \
			-DSIV3D_ROOT="${SIV3D_ROOT}" \
			-DCMAKE_BUILD_TYPE=Release

		cmake --build "${BUILD_DIR}" --parallel "${BUILD_PARALLELISM}"

		cd "${TEST_ROOT}"
		"${BUILD_DIR}/toy-acai"

		ctest --test-dir "${BUILD_DIR}" --output-on-failure
	'

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=${TOY_ACAI_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
REPO_ROOT=${TOY_ACAI_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}

SIV3D_ROOT=${SIV3D_ROOT:-"${REPO_ROOT}/../siv3d/OpenSiv3D"}
SIV3D_APPTAINER_IMAGE=${SIV3D_APPTAINER_IMAGE:-"${HOME}/container_image/siv3d-ubuntu22.sif"}
BUILD_DIR=${BUILD_DIR:-"${SCRIPT_DIR}/build"}
BUILD_PARALLELISM=${BUILD_PARALLELISM:-1}
SLURM_PARTITION=${SLURM_PARTITION:-gr10561a}
SLURM_TIME=${SLURM_TIME:-00:10:00}

if [[ ! -f "${SIV3D_APPTAINER_IMAGE}" ]]; then
	echo "Apptainer image was not found: ${SIV3D_APPTAINER_IMAGE}" >&2
	exit 1
fi

if [[ ! -f "${SIV3D_ROOT}/Linux/build/libSiv3D.a" ]]; then
	echo "libSiv3D.a was not found: ${SIV3D_ROOT}/Linux/build/libSiv3D.a" >&2
	exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" && -z "${TOY_ACAI_INSIDE_SLURM:-}" ]]; then
	mkdir -p "${BUILD_DIR}"
	set +e
	job_id=$(sbatch \
		--parsable \
		--wait \
		-p "${SLURM_PARTITION}" \
		--time="${SLURM_TIME}" \
		--job-name=toy-acai-build \
		--output="${BUILD_DIR}/slurm-%j.out" \
		--export=ALL,TOY_ACAI_INSIDE_SLURM=1,TOY_ACAI_SCRIPT_DIR="${SCRIPT_DIR}",TOY_ACAI_REPO_ROOT="${REPO_ROOT}",SIV3D_ROOT="${SIV3D_ROOT}",SIV3D_APPTAINER_IMAGE="${SIV3D_APPTAINER_IMAGE}",BUILD_DIR="${BUILD_DIR}",BUILD_PARALLELISM="${BUILD_PARALLELISM}" \
		"${BASH_SOURCE[0]}")
	status=$?
	set -e

	job_id=${job_id%%;*}
	if [[ -n "${job_id}" && -f "${BUILD_DIR}/slurm-${job_id}.out" ]]; then
		cat "${BUILD_DIR}/slurm-${job_id}.out"
	fi
	exit "${status}"
fi

echo "Running on $(hostname) via Slurm job ${SLURM_JOB_ID:-unknown}"

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

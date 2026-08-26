#!/bin/bash
# Build the S2S container and (optionally) convert it to an Apptainer image.
#
#   ./docker/build.sh                 # docker image only
#   ./docker/build.sh --sif           # also produce s2s.sif for HPC
#   ./docker/build.sh --sif --mpi     # include mpi4py for data_utils scripts
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-s2s:latest}"
SIF_PATH="${SIF_PATH:-s2s.sif}"
BUILD_SIF=false
MPI_ARG="false"

for arg in "$@"; do
  case "$arg" in
    --sif) BUILD_SIF=true ;;
    --mpi) MPI_ARG="true" ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "==> Building ${IMAGE} (WITH_MPI=${MPI_ARG})"
docker build -f docker/Dockerfile --build-arg WITH_MPI="${MPI_ARG}" -t "${IMAGE}" .

if [ "${BUILD_SIF}" = true ]; then
  echo "==> Converting ${IMAGE} -> ${SIF_PATH}"
  if command -v apptainer >/dev/null 2>&1; then
    SINGULARITY=apptainer
  elif command -v singularity >/dev/null 2>&1; then
    SINGULARITY=singularity
  else
    # No local Apptainer: fall back to converting through a docker archive.
    echo "    apptainer not found locally; writing a docker archive instead."
    echo "    Copy s2s-docker.tar to the HPC system and run there:"
    echo "      apptainer build ${SIF_PATH} docker-archive://s2s-docker.tar"
    docker save -o s2s-docker.tar "${IMAGE}"
    echo "==> Wrote s2s-docker.tar"
    exit 0
  fi
  # Go through a local docker-daemon reference so no registry push is needed.
  ${SINGULARITY} build "${SIF_PATH}" "docker-daemon://${IMAGE}"
  echo "==> Wrote ${SIF_PATH}"
fi

echo "==> Done"

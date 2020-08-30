#!/usr/bin/env bash
set -e

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <python-version> <tensorflow-version>"
  exit 1
fi

PYTHON_VERSION="$1"
TF_VERSION="$2"

IMAGE="locality-aware-nms-build"
WORKDIR="/locality-aware-nms"

docker build -t ${IMAGE} .
docker run -it --rm \
  -v $(pwd)/artifacts:"${WORKDIR}/artifacts" \
  -v $(pwd)/lanms:"${WORKDIR}/lanms" \
  -v $(pwd)/BUILD:"${WORKDIR}/BUILD" \
  -v $(pwd)/WORKSPACE:"${WORKDIR}/WORKSPACE" \
  -v $(pwd)/configure.sh:"${WORKDIR}/configure.sh" \
  -v $(pwd)/pip_package:"${WORKDIR}/pip_package" \
  ${IMAGE} bash -c "./configure.sh ${PYTHON_VERSION} ${TF_VERSION}  && bazel build build_pip_pkg && bazel-bin/build_pip_pkg artifacts"

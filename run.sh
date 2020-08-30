#!/usr/bin/env bash
set -e
set -x

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CMD"
  exit 1
fi

CMD="$1"
shift

case ${CMD} in
  dev)
    DOCKER_CMD="bash"
    ;;
  test)
    DOCKER_CMD="bazel run lanms:nms_test"
    ;;
  build)
    if [[ $# -lt 2 ]]; then  
      echo "Usage: $0 build PYTHON_VERSION TENSORFLOW_VERSION"
      exit 1
    fi

    PYTHON_VERSION="$1"
    TF_VERSION="$2"

    # TODO: Fix this and remove build.sh file.
    exit 1

    #BUILD_CMD="./configure.sh ${PYTHON_VERSION} ${TF_VERSION} && bazel build build_pip_pkg && bazel-bin/build_pip_pkg artifacts"
    #DOCKER_CMD="bash -c ""\"./configure.sh 3.6 2.3.0 && bazel build build_pip_pkg && bazel-bin/build_pip_pkg artifacts\""
    ;;
  *)
    echo "Invalid command."
    exit 1
    ;;
esac

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
  ${IMAGE} ${DOCKER_CMD}

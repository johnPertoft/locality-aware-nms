#!/usr/bin/env bash
set -e

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <python-version> <tensorflow-version>"
  exit 1
fi

PYTHON_VERSION="$1"
TF_VERSION="$2"

rm -f .bazelrc
rm -f .python_version

PYTHON="python${PYTHON_VERSION}"
PIP="pip${PYTHON_VERSION}"
echo "${PYTHON_VERSION}" >> .python_version

${PIP} install --upgrade tensorflow=="${TF_VERSION}"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"

TF_CFLAGS="$(${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')"
TF_LFLAGS="$(${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"
HEADER_DIR="${TF_CFLAGS:2}"
SHARED_LIBRARY_DIR="${TF_LFLAGS:2}"
SHARED_LIBRARY_NAME="$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)"

write_action_env_to_bazelrc "TF_NEED_CUDA" 0
write_action_env_to_bazelrc "TF_HEADER_DIR" ${HEADER_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}

#!/usr/bin/env bash
set -e


PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
function is_windows() {
  if [[ "${PLATFORM}" =~ (cygwin|mingw32|mingw64|msys)_nt* ]]; then
    true
  else
    false
  fi
}

if is_windows; then
  PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.exe.runfiles/__main__/"
else
  PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.runfiles/__main__/"
fi

function main() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0  <destination-dir>"
    exit 1
  fi

  DEST=$1

  PYTHON_VERSION="$(cat .python_version)"
  PYTHON="python${PYTHON_VERSION}"

  # Create the directory, then do dirname on a non-existent file inside it to
  # give us an absolute paths with tilde characters resolved to the destination
  # directory.
  mkdir -p ${DEST}
  if [[ ${PLATFORM} == "darwin" ]]; then
    DEST=$(pwd -P)/${DEST}
  else
    DEST=$(readlink -f "${DEST}")
  fi
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy TensorFlow Custom op files"

  cp ${PIP_FILE_PREFIX}pip_package/setup.py "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}pip_package/MANIFEST.in "${TMPDIR}"
  rsync -avm -L --exclude='*_test.py' ${PIP_FILE_PREFIX}lanms "${TMPDIR}"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  ${PYTHON} setup.py bdist_wheel > /dev/null

  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"

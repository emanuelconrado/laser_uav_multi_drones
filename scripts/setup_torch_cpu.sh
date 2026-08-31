#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIRECTORY="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"

# This script is expected at:
#   laser_uav_multi_drones/scripts/setup_torch_cpu.sh
readonly PACKAGE_DIRECTORY="$(
  cd -- "${SCRIPT_DIRECTORY}/.." >/dev/null 2>&1
  pwd
)"

readonly VENV_DIRECTORY="${PACKAGE_DIRECTORY}/.venv_torch_cpu"
readonly VENV_PYTHON="${VENV_DIRECTORY}/bin/python"
readonly PYTORCH_VERSION="2.8.0"
readonly PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 was not found." >&2
  exit 1
fi

echo "Package directory: ${PACKAGE_DIRECTORY}"
echo "Virtual environment: ${VENV_DIRECTORY}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Creating the CPU-only PyTorch virtual environment..."

  if ! python3 -m venv "${VENV_DIRECTORY}"; then
    echo >&2
    echo "Error: the Python virtual environment could not be created." >&2
    echo "On Ubuntu, install the venv support package and run this script again:" >&2
    echo "  sudo apt install python3-venv" >&2
    exit 1
  fi
else
  echo "The virtual environment already exists; reusing it."
fi

echo "Updating pip..."
"${VENV_PYTHON}" -m pip install --upgrade pip

echo "Installing PyTorch ${PYTORCH_VERSION} CPU-only..."
"${VENV_PYTHON}" -m pip install \
  "torch==${PYTORCH_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"

echo "Validating the installation..."
"${VENV_PYTHON}" - <<'PY'
import torch

expected_version = "2.8.0"
installed_version = torch.__version__.split("+")[0]

if installed_version != expected_version:
    raise RuntimeError(
        f"Expected PyTorch {expected_version}, "
        f"but found {torch.__version__}."
    )

if torch.version.cuda is not None:
    raise RuntimeError(
        "The installed PyTorch still uses CUDA: "
        f"{torch.version.cuda}"
    )

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CMake prefix: {torch.utils.cmake_prefix_path}")
PY

echo
echo "CPU-only PyTorch environment configured successfully."
echo "You can now build the package with:"
echo "  cd <ros2_workspace>"
echo "  colcon build --packages-select laser_uav_multi_drones --cmake-clean-cache"

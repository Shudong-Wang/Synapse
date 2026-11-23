#!/usr/bin/env bash
set -eo pipefail
if [[ ${CONDA_DEFAULT_ENV:-} != ML ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ML
fi
set -u
config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$config_dir/../../src${PYTHONPATH:+:$PYTHONPATH}"
exec python "$config_dir/launch.py" "$@"

#!/bin/bash

# Be strict and verbose.
set -euxo pipefail

# Move to dabl repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/.."

for py_vers in 3.{8..12}; do
    conda_env="dabl-py${py_vers}"
    # Run pytest foreach supported version of python.
    conda run -n $conda_env pytest -n auto -x --failed-first --new-first dabl/
done
echo "OK"

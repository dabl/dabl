#!/bin/bash

# Be strict and verbose.
set -euxo pipefail

# Move to dabl repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/.."

for py_vers in 3.{8..12}; do
    for sklearn_vers in {1.1.*,1.2.*,1.3.*,1.4.*,1.5.*}; do
        # Skip some unsupported version combos.
        # See Also: setup.py
        if [ $py_vers == "3.8" ] && [ $sklearn_vers == "1.4.*" ]; then
            continue
        elif [ $py_vers == "3.8" ] && [ $sklearn_vers == "1.5.*" ]; then
            continue
        elif [ $py_vers == "3.9" ] && [ $sklearn_vers == "1.1.*" ]; then
            continue
        elif [ $py_vers == "3.9" ] && [ $sklearn_vers == "1.2.*" ]; then
            continue
        elif [ $py_vers == "3.10" ] && [ $sklearn_vers == "1.1.*" ]; then
            continue
        elif [ $py_vers == "3.10" ] && [ $sklearn_vers == "1.2.*" ]; then
            continue
        elif [ $py_vers == "3.11" ] && [ $sklearn_vers == "1.1.*" ]; then
            continue
        elif [ $py_vers == "3.11" ] && [ $sklearn_vers == "1.2.*" ]; then
            continue
        elif [ $py_vers == "3.12" ] && [ $sklearn_vers == "1.1.*" ]; then
            continue
        elif [ $py_vers == "3.12" ] && [ $sklearn_vers == "1.2.*" ]; then
            continue
        fi
        # Run the tests.
        conda_env="dabl-py${py_vers}"
        conda run -n $conda_env pip install -U scikit-learn==$sklearn_vers
        conda run -n $conda_env pip check
        # Run pytest foreach supported version of python.
        conda run -n $conda_env pytest -n auto -x --failed-first --new-first dabl/
    done
done
echo "OK"

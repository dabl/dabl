#!/bin/bash

# Be strict and verbose.
set -euxo pipefail

# Move to dabl repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/.."

for py_vers in 3.{8..12}; do
    # Setup/update a conda environment for each supported Python version.
    conda create -y -n dabl-py${py_vers} python=${py_vers} || conda update -y -n dabl-py${py_vers} --all
    # Install dabl and it's dependencies.
    conda run -n dabl-py${py_vers} pip install -e .
    # Install some dev requirements.
    conda run -n dabl-py${py_vers} pip install pytest flake8
done
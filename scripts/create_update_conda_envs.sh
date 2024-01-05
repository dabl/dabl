#!/bin/bash

# Creates or updates conda environments for each supported Python version.
# NOTE: This works on latest upstream versions of dependent packages.
# Additional work is required to test older dependencies (e.g., scikit-learn==1.2.*)

# Be strict and verbose.
set -euxo pipefail

# Move to dabl repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/.."

for py_vers in 3.{8..12}; do
    conda_env="dabl-py${py_vers}"
    # Setup/update a conda environment for each supported Python version.
    conda create -y -n $conda_env python=${py_vers} || conda update -y -n $conda_env --all
    # Install dabl and it's dependencies.
    conda run -n $conda_env pip install -U -e .
    # Install some dev requirements.
    conda run -n $conda_env pip install -U pytest pytest-xdist flake8
    # Upgrade all pip packages.
    # (the goal is to keep all environments as close to the same as possible)
    # https://stackoverflow.com/questions/2720014/how-to-upgrade-all-python-packages-with-pip
    conda run -n $conda_env pip list --outdated --format=columns \
        | awk '( NR > 2 ) { print $1 }' \
        | xargs -r conda run -n $conda_env pip install -U
done

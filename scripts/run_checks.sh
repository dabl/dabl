#!/bin/bash

# Be strict and verbose.
set -euxo pipefail

# Move to dabl repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/.."

for py_vers in 3.{8..12}; do
    # Run flake8 foreach supported version of python.
    conda run -n dabl-py${py_vers} flake8 dabl/
    # TODO: run pylint?
done
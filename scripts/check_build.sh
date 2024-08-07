#!/bin/bash

# Be strict and verbose.
set -euxo pipefail

# Move to dabl repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/.."

rm -rf build/ dist/
python -m build || python3 -m build
# Basic validation of the whl produced.
unzip -l dist/dabl-*-py3-none-any.whl | grep dabl/__init__.py
# Make sure tests and docs aren't included.
! (unzip -l dist/dabl-*-py3-none-any.whl | grep dabl/test )
! (unzip -l dist/dabl-*-py3-none-any.whl | grep dabl/doc )
# Make sure the package metadata looks right.
unzip -p dist/dabl-*-py3-none-any.whl dabl-*/METADATA | grep 'pip install dabl'
# Make sure the source tar looks OK.
tar tzvf dist/dabl-*.tar.gz | grep portfolios/

echo "OK"

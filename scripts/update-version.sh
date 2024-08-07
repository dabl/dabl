#!/bin/bash

# This script is used to update the git tags, version numbers, etc. in a number of files.
# See Also: .bumpversion.cfg (which gets rewritten by the tool, and strips comments, so we keep a separate config file for it)

set -eu

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/.."

set -x

# Example usage: "./update-version.sh --dry-run patch" to bump v0.0.4 -> v0.0.5, for instance.
# Example usage: "./update-version.sh --dry-run minor" to bump v0.0.4 -> v0.1.0, for instance.

# Note: the tag generated locally can be used for testing, but needs to reset
# to the upstream commit once the PR to bump the version is merged so that a
# new release on Github can be created.
#
python3 -m pip install bump2version
bumpversion --verbose $*

#!/usr/bin/env bash

################################################################################
# Fetches the latest version of cirq from github, transpiles it to python 2, and
# installs the transpiled code into the current virtual environment.
#
# Usage:
#     dev_tools/pip-install-cirq-python2-from-head.sh
#
# You must be in a python 2 environment to run this command. 3to2 will be
# installed in order to perform the transpilation.
################################################################################

set -e

# Get the working directory to the repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
cd $(git rev-parse --show-toplevel)

# Do a sanity check that we're actually in python 2 land.
v=$(python --version 2>&1)
h=$(echo "${v}" | grep "Python 2\.7\.")
if [ -z "${h}" ]; then
    echo -e "\e[31mError: 'python --version' was '${v}', not 'Python 2.7.*'.\nDouble-check that you are in a prepared python 2 environment.\e[0m"
    exit 1
fi

# Prepare work area.
tmp2_dir=$(mktemp -d "/tmp/cirq2.head.XXXXXXXXXXXXXXXX")
tmp3_dir=$(mktemp -d "/tmp/cirq3.head.XXXXXXXXXXXXXXXX")
rmdir "${tmp2_dir}"
rmdir "${tmp3_dir}"
trap "{ rm -rf ${tmp2_dir}; }" EXIT
trap "{ rm -rf ${tmp3_dir}; }" EXIT
trap "exit" INT

# Get python 3 code, convert to python 2, then install it.
git clone git://github.com/quantumlib/Cirq.git "${tmp3_dir}" --depth=1 --quiet
pip install 3to2
echo "Transpiling cirq to python 2.7..."
bash "${tmp3_dir}/dev_tools/python2.7-generate.sh" "${tmp2_dir}" "${tmp3_dir}"
pip install "${tmp2_dir}"

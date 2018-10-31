# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from setuptools import find_packages, setup

# This reads the __version__ variable from openfermioncirq/_version.py
__version__ = None
exec(open('openfermioncirq/_version.py').read())

# Readme file as long_description:
long_description = ('================\n' +
                    'OpenFermion-Cirq\n' +
                    '================\n')
stream = io.open('README.rst', encoding='utf-8')
stream.readline()
long_description += stream.read()
description = ('Quantum circuits for simulations ' +
               'of quantum chemistry and materials.')

# Read in runtime-requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

openfermioncirq_packages = ['openfermioncirq'] + [
    'openfermioncirq.' + package
    for package in find_packages(where='openfermioncirq')
]

setup(
    name='openfermioncirq',
    version=__version__,
    url='https://github.com/quantumlib/OpenFermion-Cirq',
    author='The OpenFermion Developers',
    author_email='openfermioncirq@googlegroups.com',
    python_requires='>=3.5.2',
    install_requires=requirements,
    license='Apache 2',
    description=description,
    long_description=long_description,
    packages=openfermioncirq_packages)

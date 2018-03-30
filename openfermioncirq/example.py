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

import cirq
import openfermion


def make_circuit_diagram():
    q = cirq.NamedQubit('q')
    r = cirq.NamedQubit('r')
    c = cirq.Circuit.from_ops(
        cirq.X(q),
        cirq.CZ(q, r)
    )
    return str(c)


def make_qubit_operator_str():
    a = openfermion.QubitOperator('X0 Z5')
    b = openfermion.QubitOperator('X5 Z7')
    return str(a * b)


def cause_failure():
    raise ValueError('failure')

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy

import cirq
import openfermion

from openfermioncirq.trotter import SPLIT_OPERATOR


n_qubits = 4
qubits = cirq.LineQubit.range(n_qubits)
control = cirq.LineQubit(-1)
ones_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
        one_body=numpy.ones((n_qubits, n_qubits)),
        two_body=numpy.ones((n_qubits, n_qubits)),
        constant=1.0)


def test_split_operator_trotter_step_symmetric():
    circuit = cirq.Circuit.from_ops(
            SPLIT_OPERATOR.symmetric(ones_hamiltonian).trotter_step(
                qubits, 1.0),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert circuit.to_text_diagram(transpose=True).strip() == """
0        1          2           3
│        │          │           │
Z^-0.159 Z^-0.159   Z^-0.159    Z^-0.796
│        │          │           │
Z        Z^0.0      Z^0.0       Z^0.0
│        │          │           │
│        │          YXXY────────#2^-1
│        │          │           │
│        YXXY───────#2^-0.333   Z^0.0
│        │          │           │
YXXY─────#2^-1      Z^0.0       │
│        │          │           │
│        Z^0.0      YXXY────────#2^-0.301
│        │          │           │
│        YXXY───────#2^0.449    Z^0.0
│        │          │           │
@────────@^-0.637   Z^0.0       │
│        │          │           │
×────────×          YXXY────────#2^-0.123
│        │          │           │
│        │          │           Z^0.0
│        │          │           │
│        │          @───────────@^-0.637
│        │          │           │
│        │          ×───────────×
│        │          │           │
│        @──────────@^-0.637    │
│        │          │           │
│        ×──────────×           │
│        │          │           │
@────────@^-0.637   @───────────@^-0.637
│        │          │           │
×────────×          ×───────────×
│        │          │           │
Z^0.0    @──────────@^-0.637    │
│        │          │           │
│        ×──────────×           │
│        │          │           │
#2───────YXXY^0.123 │           │
│        │          │           │
Z^0.0    Z^0.0      │           │
│        │          │           │
│        #2─────────YXXY^-0.449 │
│        │          │           │
#2───────YXXY^0.301 Z^0.0       │
│        │          │           │
Z^0.0    Z^0.0      #2──────────YXXY
│        │          │           │
│        #2─────────YXXY^0.333  Z^-1
│        │          │           │
#2───────YXXY       Z^0.0       Z^-0.159
│        │          │           │
Z^0.0    Z^0.0      Z^-0.159    │
│        │          │           │
Z^-0.796 Z^-0.159   │           │
│        │          │           │
""".strip()


def test_split_operator_trotter_step_controlled_symmetric():
    circuit = cirq.Circuit.from_ops(
            SPLIT_OPERATOR.controlled_symmetric(ones_hamiltonian).trotter_step(
                qubits, 1.0, control),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert circuit.to_text_diagram(transpose=True).strip() == """
  -1       0        1          2           3
  │        │        │          │           │
  @────────@^-0.159 │          │           │
┌ │        │        │          │           │         ┐
│ @────────┼────────@^-0.159   │           │         │
│ │        Z        │          │           │         │
└ │        │        │          │           │         ┘
┌ │        │        │          │           │         ┐
│ @────────┼────────┼──────────@^-0.159    │         │
│ │        │        Z^0.0      │           │         │
└ │        │        │          │           │         ┘
┌ │        │        │          │           │         ┐
│ @────────┼────────┼──────────┼───────────@^-0.796  │
│ │        │        │          Z^0.0       │         │
└ │        │        │          │           │         ┘
  │        │        │          │           Z^0.0
  │        │        │          │           │
  │        │        │          YXXY────────#2^-1
  │        │        │          │           │
  │        │        YXXY───────#2^-0.333   Z^0.0
  │        │        │          │           │
  │        YXXY─────#2^-1      Z^0.0       │
  │        │        │          │           │
  │        │        Z^0.0      YXXY────────#2^-0.301
  │        │        │          │           │
  │        │        YXXY───────#2^0.449    Z^0.0
  │        │        │          │           │
  @────────@────────@^-0.637   Z^0.0       │
  │        │        │          │           │
  │        ×────────×          YXXY────────#2^-0.123
  │        │        │          │           │
  │        │        │          │           Z^0.0
  │        │        │          │           │
  @────────┼────────┼──────────@───────────@^-0.637
  │        │        │          │           │
  │        │        │          ×───────────×
  │        │        │          │           │
  @────────┼────────@──────────@^-0.637    │
  │        │        │          │           │
  │        │        ×──────────×           │
  │        │        │          │           │
  @────────@────────@^-0.637   │           │
┌ │        │        │          │           │         ┐
│ │        ×────────×          │           │         │
│ @────────┼────────┼──────────@───────────@^-0.637  │
└ │        │        │          │           │         ┘
  │        Z^0.0    │          ×───────────×
  │        │        │          │           │
  @────────┼────────@──────────@^-0.637    │
  │        │        │          │           │
  │        │        ×──────────×           │
  │        │        │          │           │
  │        #2───────YXXY^0.123 │           │
  │        │        │          │           │
  │        Z^0.0    Z^0.0      │           │
  │        │        │          │           │
  │        │        #2─────────YXXY^-0.449 │
  │        │        │          │           │
  │        #2───────YXXY^0.301 Z^0.0       │
  │        │        │          │           │
  │        Z^0.0    Z^0.0      #2──────────YXXY
  │        │        │          │           │
  │        │        #2─────────YXXY^0.333  Z^-1
┌ │        │        │          │           │         ┐
│ │        #2───────YXXY       Z^0.0       │         │
│ @────────┼────────┼──────────┼───────────@^-0.159  │
└ │        │        │          │           │         ┘
┌ │        │        │          │           │         ┐
│ │        Z^0.0    Z^0.0      │           │         │
│ @────────┼────────┼──────────@^-0.159    │         │
└ │        │        │          │           │         ┘
  @────────┼────────@^-0.159   │           │
  │        │        │          │           │
  @────────@^-0.796 │          │           │
  │        │        │          │           │
  Z^-0.318 │        │          │           │
  │        │        │          │           │
""".strip()


def test_split_operator_trotter_step_asymmetric():
    circuit = cirq.Circuit.from_ops(
            SPLIT_OPERATOR.asymmetric(ones_hamiltonian).trotter_step(
                qubits, 1.0),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert circuit.to_text_diagram(transpose=True).strip() == """
0       1           2           3
│       │           │           │
@───────@^-0.637    @───────────@^-0.637
│       │           │           │
×───────×           ×───────────×
│       │           │           │
│       @───────────@^-0.637    │
│       │           │           │
│       ×───────────×           │
│       │           │           │
@───────@^-0.637    @───────────@^-0.637
│       │           │           │
×───────×           ×───────────×
│       │           │           │
Z^0.0   @───────────@^-0.637    │
│       │           │           │
│       ×───────────×           │
│       │           │           │
#2──────YXXY^0.123  │           │
│       │           │           │
Z^0.0   Z^0.0       │           │
│       │           │           │
│       #2──────────YXXY^-0.449 │
│       │           │           │
#2──────YXXY^0.301  Z^0.0       │
│       │           │           │
Z^0.0   Z^0.0       #2──────────YXXY
│       │           │           │
│       #2──────────YXXY^0.333  Z^-1
│       │           │           │
#2──────YXXY        Z^0.0       Z^-0.318
│       │           │           │
Z^0.0   Z^0.0       Z^-0.318    Z
│       │           │           │
Z^-1.59 Z^-0.318    Z^0.0       │
│       │           │           │
Z^0.0   Z^0.0       │           │
│       │           │           │
#2──────YXXY^-1     │           │
│       │           │           │
Z^0.0   #2──────────YXXY^-0.333 │
│       │           │           │
│       Z^0.0       #2──────────YXXY^-1
│       │           │           │
#2──────YXXY^-0.301 Z^0.0       │
│       │           │           │
Z^0.0   #2──────────YXXY^0.449  │
│       │           │           │
│       Z^0.0       │           │
│       │           │           │
#2──────YXXY^-0.123 │           │
│       │           │           │
Z^0.0   │           │           │
│       │           │           │
""".strip()


def test_split_operator_trotter_step_controlled_asymmetric():
    circuit = cirq.Circuit.from_ops(
            SPLIT_OPERATOR.controlled_asymmetric(ones_hamiltonian).trotter_step(
                qubits, 1.0, control),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert circuit.to_text_diagram(transpose=True).strip() == """
  -1       0       1           2           3
  │        │       │           │           │
  @────────@───────@^-0.637    │           │
┌ │        │       │           │           │        ┐
│ │        ×───────×           │           │        │
│ @────────┼───────┼───────────@───────────@^-0.637 │
└ │        │       │           │           │        ┘
  │        │       │           ×───────────×
  │        │       │           │           │
  @────────┼───────@───────────@^-0.637    │
  │        │       │           │           │
  │        │       ×───────────×           │
  │        │       │           │           │
  @────────@───────@^-0.637    │           │
┌ │        │       │           │           │        ┐
│ │        ×───────×           │           │        │
│ @────────┼───────┼───────────@───────────@^-0.637 │
└ │        │       │           │           │        ┘
  │        Z^0.0   │           ×───────────×
  │        │       │           │           │
  @────────┼───────@───────────@^-0.637    │
  │        │       │           │           │
  │        │       ×───────────×           │
  │        │       │           │           │
  │        #2──────YXXY^0.123  │           │
  │        │       │           │           │
  │        Z^0.0   Z^0.0       │           │
  │        │       │           │           │
  │        │       #2──────────YXXY^-0.449 │
  │        │       │           │           │
  │        #2──────YXXY^0.301  Z^0.0       │
  │        │       │           │           │
  │        Z^0.0   Z^0.0       #2──────────YXXY
  │        │       │           │           │
  │        │       #2──────────YXXY^0.333  Z^-1
┌ │        │       │           │           │        ┐
│ │        #2──────YXXY        Z^0.0       │        │
│ @────────┼───────┼───────────┼───────────@^-0.318 │
└ │        │       │           │           │        ┘
  │        Z^0.0   Z^0.0       │           Z
  @────────┼───────┼───────────@^-0.318    │
  │        │       │           │           │
  @────────┼───────@^-0.318    Z^0.0       │
  │        │       │           │           │
  @────────@^0.408 Z^0.0       │           │
  │        │       │           │           │
  Z^-0.318 Z^0.0   │           │           │
  │        │       │           │           │
  │        #2──────YXXY^-1     │           │
  │        │       │           │           │
  │        Z^0.0   #2──────────YXXY^-0.333 │
  │        │       │           │           │
  │        │       Z^0.0       #2──────────YXXY^-1
  │        │       │           │           │
  │        #2──────YXXY^-0.301 Z^0.0       │
  │        │       │           │           │
  │        Z^0.0   #2──────────YXXY^0.449  │
  │        │       │           │           │
  │        │       Z^0.0       │           │
  │        │       │           │           │
  │        #2──────YXXY^-0.123 │           │
  │        │       │           │           │
  │        Z^0.0   │           │           │
  │        │       │           │           │
""".strip()

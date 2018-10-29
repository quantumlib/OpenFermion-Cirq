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

from openfermioncirq.trotter import LINEAR_SWAP_NETWORK


n_qubits = 4
qubits = cirq.LineQubit.range(n_qubits)
control = cirq.LineQubit(-1)
ones_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
        one_body=numpy.ones((n_qubits, n_qubits)),
        two_body=numpy.ones((n_qubits, n_qubits)),
        constant=1.0)


def test_linear_swap_network_trotter_step_symmetric():
    circuit = cirq.Circuit.from_ops(
            LINEAR_SWAP_NETWORK.symmetric(
                ones_hamiltonian).trotter_step(qubits, 1.0),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert circuit.to_text_diagram(transpose=True).strip() == """
0        1          2          3
│        │          │          │
XXYY─────XXYY^0.318 XXYY───────XXYY^0.318
│        │          │          │
YXXY─────#2^0.0     YXXY───────#2^0.0
│        │          │          │
@────────@^-0.318   @──────────@^-0.318
│        │          │          │
×ᶠ───────×ᶠ         ×ᶠ─────────×ᶠ
│        │          │          │
│        XXYY───────XXYY^0.318 │
│        │          │          │
│        YXXY───────#2^0.0     │
│        │          │          │
│        @──────────@^-0.318   │
│        │          │          │
│        ×ᶠ─────────×ᶠ         │
│        │          │          │
XXYY─────XXYY^0.318 XXYY───────XXYY^0.318
│        │          │          │
YXXY─────#2^0.0     YXXY───────#2^0.0
│        │          │          │
@────────@^-0.318   @──────────@^-0.318
│        │          │          │
×ᶠ───────×ᶠ         ×ᶠ─────────×ᶠ
│        │          │          │
Z^-0.637 XXYY───────XXYY^0.318 Z^-0.637
│        │          │          │
│        YXXY───────#2^0.0     │
│        │          │          │
│        @──────────@^-0.318   │
│        │          │          │
│        ×ᶠ─────────×ᶠ         │
│        │          │          │
│        Z^-0.637   Z^-0.637   │
│        │          │          │
│        @──────────@^-0.318   │
│        │          │          │
│        #2─────────YXXY^0.0   │
│        │          │          │
│        XXYY───────XXYY^0.318 │
│        │          │          │
│        ×ᶠ─────────×ᶠ         │
│        │          │          │
@────────@^-0.318   @──────────@^-0.318
│        │          │          │
#2───────YXXY^0.0   #2─────────YXXY^0.0
│        │          │          │
XXYY─────XXYY^0.318 XXYY───────XXYY^0.318
│        │          │          │
×ᶠ───────×ᶠ         ×ᶠ─────────×ᶠ
│        │          │          │
│        @──────────@^-0.318   │
│        │          │          │
│        #2─────────YXXY^0.0   │
│        │          │          │
│        XXYY───────XXYY^0.318 │
│        │          │          │
│        ×ᶠ─────────×ᶠ         │
│        │          │          │
@────────@^-0.318   @──────────@^-0.318
│        │          │          │
#2───────YXXY^0.0   #2─────────YXXY^0.0
│        │          │          │
XXYY─────XXYY^0.318 XXYY───────XXYY^0.318
│        │          │          │
×ᶠ───────×ᶠ         ×ᶠ─────────×ᶠ
│        │          │          │
""".strip()


def test_linear_swap_network_trotter_step_controlled_symmetric():
    circuit = cirq.Circuit.from_ops(
            LINEAR_SWAP_NETWORK.controlled_symmetric(
                ones_hamiltonian).trotter_step(
                qubits, 1.0, control),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert circuit.to_text_diagram(transpose=True).strip() == """
  -1       0        1          2          3
  │        │        │          │          │
  @────────XXYY─────XXYY^0.318 │          │
  │        │        │          │          │
  @────────YXXY─────#2^0.0     │          │
  │        │        │          │          │
  @────────@────────@^-0.318   │          │
┌ │        │        │          │          │          ┐
│ │        ×ᶠ───────×ᶠ         │          │          │
│ @────────┼────────┼──────────XXYY───────XXYY^0.318 │
└ │        │        │          │          │          ┘
  @────────┼────────┼──────────YXXY───────#2^0.0
  │        │        │          │          │
  @────────┼────────┼──────────@──────────@^-0.318
  │        │        │          │          │
  │        │        │          ×ᶠ─────────×ᶠ
  │        │        │          │          │
  @────────┼────────XXYY───────XXYY^0.318 │
  │        │        │          │          │
  @────────┼────────YXXY───────#2^0.0     │
  │        │        │          │          │
  @────────┼────────@──────────@^-0.318   │
  │        │        │          │          │
  │        │        ×ᶠ─────────×ᶠ         │
  │        │        │          │          │
  @────────XXYY─────XXYY^0.318 │          │
  │        │        │          │          │
  @────────YXXY─────#2^0.0     │          │
  │        │        │          │          │
  @────────@────────@^-0.318   │          │
┌ │        │        │          │          │          ┐
│ │        ×ᶠ───────×ᶠ         │          │          │
│ @────────┼────────┼──────────XXYY───────XXYY^0.318 │
└ │        │        │          │          │          ┘
  @────────┼────────┼──────────YXXY───────#2^0.0
  │        │        │          │          │
  @────────┼────────┼──────────@──────────@^-0.318
  │        │        │          │          │
  │        │        │          ×ᶠ─────────×ᶠ
  │        │        │          │          │
  @────────┼────────XXYY───────XXYY^0.318 │
  │        │        │          │          │
  @────────┼────────YXXY───────#2^0.0     │
  │        │        │          │          │
  @────────┼────────@──────────@^-0.318   │
┌ │        │        │          │          │          ┐
│ │        │        ×ᶠ─────────×ᶠ         │          │
│ @────────┼────────┼──────────┼──────────@^-0.637   │
└ │        │        │          │          │          ┘
  @────────┼────────┼──────────@^-0.637   │
  │        │        │          │          │
  @────────┼────────@^-0.637   │          │
  │        │        │          │          │
  @────────@^-0.637 │          │          │
  │        │        │          │          │
  @────────┼────────@──────────@^-0.318   │
  │        │        │          │          │
  @────────┼────────#2─────────YXXY^0.0   │
  │        │        │          │          │
  @────────┼────────XXYY───────XXYY^0.318 │
  │        │        │          │          │
  │        │        ×ᶠ─────────×ᶠ         │
  │        │        │          │          │
  @────────┼────────┼──────────@──────────@^-0.318
  │        │        │          │          │
  @────────┼────────┼──────────#2─────────YXXY^0.0
  │        │        │          │          │
  @────────┼────────┼──────────XXYY───────XXYY^0.318
  │        │        │          │          │
  @────────@────────@^-0.318   ×ᶠ─────────×ᶠ
  │        │        │          │          │
  @────────#2───────YXXY^0.0   │          │
  │        │        │          │          │
  @────────XXYY─────XXYY^0.318 │          │
  │        │        │          │          │
  │        ×ᶠ───────×ᶠ         │          │
  │        │        │          │          │
  @────────┼────────@──────────@^-0.318   │
  │        │        │          │          │
  @────────┼────────#2─────────YXXY^0.0   │
  │        │        │          │          │
  @────────┼────────XXYY───────XXYY^0.318 │
  │        │        │          │          │
  │        │        ×ᶠ─────────×ᶠ         │
  │        │        │          │          │
  @────────┼────────┼──────────@──────────@^-0.318
  │        │        │          │          │
  @────────┼────────┼──────────#2─────────YXXY^0.0
  │        │        │          │          │
  @────────┼────────┼──────────XXYY───────XXYY^0.318
  │        │        │          │          │
  @────────@────────@^-0.318   ×ᶠ─────────×ᶠ
  │        │        │          │          │
  @────────#2───────YXXY^0.0   │          │
  │        │        │          │          │
  @────────XXYY─────XXYY^0.318 │          │
  │        │        │          │          │
  Z^-0.318 ×ᶠ───────×ᶠ         │          │
  │        │        │          │          │
""".strip()


def test_linear_swap_network_trotter_step_asymmetric():
    circuit = cirq.Circuit.from_ops(
            LINEAR_SWAP_NETWORK.asymmetric(
                ones_hamiltonian).trotter_step(qubits, 1.0),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert circuit.to_text_diagram(transpose=True).strip() == """
0        1          2          3
│        │          │          │
XXYY─────XXYY^0.637 XXYY───────XXYY^0.637
│        │          │          │
YXXY─────#2^0.0     YXXY───────#2^0.0
│        │          │          │
@────────@^-0.637   @──────────@^-0.637
│        │          │          │
×ᶠ───────×ᶠ         ×ᶠ─────────×ᶠ
│        │          │          │
│        XXYY───────XXYY^0.637 │
│        │          │          │
│        YXXY───────#2^0.0     │
│        │          │          │
│        @──────────@^-0.637   │
│        │          │          │
│        ×ᶠ─────────×ᶠ         │
│        │          │          │
XXYY─────XXYY^0.637 XXYY───────XXYY^0.637
│        │          │          │
YXXY─────#2^0.0     YXXY───────#2^0.0
│        │          │          │
@────────@^-0.637   @──────────@^-0.637
│        │          │          │
×ᶠ───────×ᶠ         ×ᶠ─────────×ᶠ
│        │          │          │
Z^-0.637 XXYY───────XXYY^0.637 Z^-0.637
│        │          │          │
│        YXXY───────#2^0.0     │
│        │          │          │
│        @──────────@^-0.637   │
│        │          │          │
│        ×ᶠ─────────×ᶠ         │
│        │          │          │
│        Z^-0.637   Z^-0.637   │
│        │          │          │
""".strip()


def test_linear_swap_network_trotter_step_controlled_asymmetric():
    circuit = cirq.Circuit.from_ops(
            LINEAR_SWAP_NETWORK.controlled_asymmetric(
                ones_hamiltonian).trotter_step(
                qubits, 1.0, control),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert circuit.to_text_diagram(transpose=True).strip() == """
  -1       0        1          2          3
  │        │        │          │          │
  @────────XXYY─────XXYY^0.637 │          │
  │        │        │          │          │
  @────────YXXY─────#2^0.0     │          │
  │        │        │          │          │
  @────────@────────@^-0.637   │          │
┌ │        │        │          │          │          ┐
│ │        ×ᶠ───────×ᶠ         │          │          │
│ @────────┼────────┼──────────XXYY───────XXYY^0.637 │
└ │        │        │          │          │          ┘
  @────────┼────────┼──────────YXXY───────#2^0.0
  │        │        │          │          │
  @────────┼────────┼──────────@──────────@^-0.637
  │        │        │          │          │
  │        │        │          ×ᶠ─────────×ᶠ
  │        │        │          │          │
  @────────┼────────XXYY───────XXYY^0.637 │
  │        │        │          │          │
  @────────┼────────YXXY───────#2^0.0     │
  │        │        │          │          │
  @────────┼────────@──────────@^-0.637   │
  │        │        │          │          │
  │        │        ×ᶠ─────────×ᶠ         │
  │        │        │          │          │
  @────────XXYY─────XXYY^0.637 │          │
  │        │        │          │          │
  @────────YXXY─────#2^0.0     │          │
  │        │        │          │          │
  @────────@────────@^-0.637   │          │
┌ │        │        │          │          │          ┐
│ │        ×ᶠ───────×ᶠ         │          │          │
│ @────────┼────────┼──────────XXYY───────XXYY^0.637 │
└ │        │        │          │          │          ┘
  @────────┼────────┼──────────YXXY───────#2^0.0
  │        │        │          │          │
  @────────┼────────┼──────────@──────────@^-0.637
  │        │        │          │          │
  │        │        │          ×ᶠ─────────×ᶠ
  │        │        │          │          │
  @────────┼────────XXYY───────XXYY^0.637 │
  │        │        │          │          │
  @────────┼────────YXXY───────#2^0.0     │
  │        │        │          │          │
  @────────┼────────@──────────@^-0.637   │
┌ │        │        │          │          │          ┐
│ │        │        ×ᶠ─────────×ᶠ         │          │
│ @────────┼────────┼──────────┼──────────@^-0.637   │
└ │        │        │          │          │          ┘
  @────────┼────────┼──────────@^-0.637   │
  │        │        │          │          │
  @────────┼────────@^-0.637   │          │
  │        │        │          │          │
  @────────@^-0.637 │          │          │
  │        │        │          │          │
  Z^-0.318 │        │          │          │
  │        │        │          │          │
""".strip()

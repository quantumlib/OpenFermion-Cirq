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
    cirq.testing.assert_has_diagram(circuit, """
0           1           2           3
│           │           │           │
Rz(-0.159π) Rz(-0.159π) Rz(-0.159π) Rz(-0.796π)
│           │           │           │
Rz(π)       Rz(0.0π)    Rz(0.0π)    Rz(0.0π)
│           │           │           │
│           │           YXXY────────#2^-1
│           │           │           │
│           YXXY────────#2^(-1/3)   Z^(0)
│           │           │           │
YXXY────────#2^-1       Z^(0)       │
│           │           │           │
│           Z^(0)       YXXY────────#2^-0.301
│           │           │           │
│           YXXY────────#2^0.449    Z^(0)
│           │           │           │
@───────────@^(-7/11)   Z^(0)       │
│           │           │           │
×───────────×           YXXY────────#2^-0.123
│           │           │           │
│           │           │           Z^(0)
│           │           │           │
│           │           @───────────@^(-7/11)
│           │           │           │
│           │           ×───────────×
│           │           │           │
│           @───────────@^(-7/11)   │
│           │           │           │
│           ×───────────×           │
│           │           │           │
@───────────@^(-7/11)   @───────────@^(-7/11)
│           │           │           │
×───────────×           ×───────────×
│           │           │           │
Z^(0)       @───────────@^(-7/11)   │
│           │           │           │
│           ×───────────×           │
│           │           │           │
#2──────────YXXY^0.123  │           │
│           │           │           │
Z^(0)       Z^(0)       │           │
│           │           │           │
│           #2──────────YXXY^-0.449 │
│           │           │           │
#2──────────YXXY^0.301  Z^(0)       │
│           │           │           │
Z^(0)       Z^(0)       #2──────────YXXY
│           │           │           │
│           #2──────────YXXY^(1/3)  Rz(-π)
│           │           │           │
#2──────────YXXY        Rz(-0.0π)   Rz(-0.159π)
│           │           │           │
Rz(-0.0π)   Rz(-0.0π)   Rz(-0.159π) │
│           │           │           │
Rz(-0.796π) Rz(-0.159π) │           │
│           │           │           │
""", transpose=True)


def test_split_operator_trotter_step_controlled_symmetric():
    circuit = cirq.Circuit.from_ops(
            SPLIT_OPERATOR.controlled_symmetric(ones_hamiltonian).trotter_step(
                qubits, 1.0, control),
            strategy=cirq.InsertStrategy.EARLIEST)
    cirq.testing.assert_has_diagram(circuit, """
  -1          0         1          2           3
  │           │         │          │           │
  @───────────@^-0.159  │          │           │
┌ │           │         │          │           │         ┐
│ @───────────┼─────────@^-0.159   │           │         │
│ │           Rz(π)     │          │           │         │
└ │           │         │          │           │         ┘
┌ │           │         │          │           │         ┐
│ @───────────┼─────────┼──────────@^-0.159    │         │
│ │           │         Rz(0.0π)   │           │         │
└ │           │         │          │           │         ┘
┌ │           │         │          │           │         ┐
│ @───────────┼─────────┼──────────┼───────────@^-0.796  │
│ │           │         │          Rz(0.0π)    │         │
└ │           │         │          │           │         ┘
  │           │         │          │           Rz(0.0π)
  │           │         │          │           │
  │           │         │          YXXY────────#2^-1
  │           │         │          │           │
  │           │         YXXY───────#2^(-1/3)   Z^(0)
  │           │         │          │           │
  │           YXXY──────#2^-1      Z^(0)       │
  │           │         │          │           │
  │           │         Z^(0)      YXXY────────#2^-0.301
  │           │         │          │           │
  │           │         YXXY───────#2^0.449    Z^(0)
  │           │         │          │           │
  @───────────@─────────@^(-7/11)  Z^(0)       │
  │           │         │          │           │
  │           ×─────────×          YXXY────────#2^-0.123
  │           │         │          │           │
  │           │         │          │           Z^(0)
  │           │         │          │           │
  @───────────┼─────────┼──────────@───────────@^(-7/11)
  │           │         │          │           │
  │           │         │          ×───────────×
  │           │         │          │           │
  @───────────┼─────────@──────────@^(-7/11)   │
  │           │         │          │           │
  │           │         ×──────────×           │
  │           │         │          │           │
  @───────────@─────────@^(-7/11)  │           │
┌ │           │         │          │           │         ┐
│ │           ×─────────×          │           │         │
│ @───────────┼─────────┼──────────@───────────@^(-7/11) │
└ │           │         │          │           │         ┘
  │           Z^(0)     │          ×───────────×
  │           │         │          │           │
  @───────────┼─────────@──────────@^(-7/11)   │
  │           │         │          │           │
  │           │         ×──────────×           │
  │           │         │          │           │
  │           #2────────YXXY^0.123 │           │
  │           │         │          │           │
  │           Z^(0)     Z^(0)      │           │
  │           │         │          │           │
  │           │         #2─────────YXXY^-0.449 │
  │           │         │          │           │
  │           #2────────YXXY^0.301 Z^(0)       │
  │           │         │          │           │
  │           Z^(0)     Z^(0)      #2──────────YXXY
  │           │         │          │           │
  │           │         #2─────────YXXY^(1/3)  Rz(-π)
┌ │           │         │          │           │         ┐
│ │           #2────────YXXY       Rz(-0.0π)   │         │
│ @───────────┼─────────┼──────────┼───────────@^-0.159  │
└ │           │         │          │           │         ┘
┌ │           │         │          │           │         ┐
│ │           Rz(-0.0π) Rz(-0.0π)  │           │         │
│ @───────────┼─────────┼──────────@^-0.159    │         │
└ │           │         │          │           │         ┘
  @───────────┼─────────@^-0.159   │           │
  │           │         │          │           │
  @───────────@^-0.796  │          │           │
  │           │         │          │           │
  Rz(-0.318π) │         │          │           │
  │           │         │          │           │
""", transpose=True)


def test_split_operator_trotter_step_asymmetric():
    circuit = cirq.Circuit.from_ops(
            SPLIT_OPERATOR.asymmetric(ones_hamiltonian).trotter_step(
                qubits, 1.0),
            strategy=cirq.InsertStrategy.EARLIEST)
    cirq.testing.assert_has_diagram(circuit, """
0           1           2           3
│           │           │           │
@───────────@^(-7/11)   @───────────@^(-7/11)
│           │           │           │
×───────────×           ×───────────×
│           │           │           │
│           @───────────@^(-7/11)   │
│           │           │           │
│           ×───────────×           │
│           │           │           │
@───────────@^(-7/11)   @───────────@^(-7/11)
│           │           │           │
×───────────×           ×───────────×
│           │           │           │
Z^(0)       @───────────@^(-7/11)   │
│           │           │           │
│           ×───────────×           │
│           │           │           │
#2──────────YXXY^0.123  │           │
│           │           │           │
Z^(0)       Z^(0)       │           │
│           │           │           │
│           #2──────────YXXY^-0.449 │
│           │           │           │
#2──────────YXXY^0.301  Z^(0)       │
│           │           │           │
Z^(0)       Z^(0)       #2──────────YXXY
│           │           │           │
│           #2──────────YXXY^(1/3)  Rz(-π)
│           │           │           │
#2──────────YXXY        Rz(-0.0π)   Rz(-0.318π)
│           │           │           │
Rz(-0.0π)   Rz(-0.0π)   Rz(-0.318π) Rz(π)
│           │           │           │
Rz(-1.592π) Rz(-0.318π) Rz(0.0π)    │
│           │           │           │
Rz(0.0π)    Rz(0.0π)    │           │
│           │           │           │
#2──────────YXXY^-1     │           │
│           │           │           │
Z^(0)       #2──────────YXXY^(-1/3) │
│           │           │           │
│           Z^(0)       #2──────────YXXY^-1
│           │           │           │
#2──────────YXXY^-0.301 Z^(0)       │
│           │           │           │
Z^(0)       #2──────────YXXY^0.449  │
│           │           │           │
│           Z^(0)       │           │
│           │           │           │
#2──────────YXXY^-0.123 │           │
│           │           │           │
Z^(0)       │           │           │
│           │           │           │
""", transpose=True)


def test_split_operator_trotter_step_controlled_asymmetric():
    circuit = cirq.Circuit.from_ops(
            SPLIT_OPERATOR.controlled_asymmetric(ones_hamiltonian).trotter_step(
                qubits, 1.0, control),
            strategy=cirq.InsertStrategy.EARLIEST)
    cirq.testing.assert_has_diagram(circuit, """
  -1          0         1           2           3
  │           │         │           │           │
  @───────────@─────────@^(-7/11)   │           │
┌ │           │         │           │           │         ┐
│ │           ×─────────×           │           │         │
│ @───────────┼─────────┼───────────@───────────@^(-7/11) │
└ │           │         │           │           │         ┘
  │           │         │           ×───────────×
  │           │         │           │           │
  @───────────┼─────────@───────────@^(-7/11)   │
  │           │         │           │           │
  │           │         ×───────────×           │
  │           │         │           │           │
  @───────────@─────────@^(-7/11)   │           │
┌ │           │         │           │           │         ┐
│ │           ×─────────×           │           │         │
│ @───────────┼─────────┼───────────@───────────@^(-7/11) │
└ │           │         │           │           │         ┘
  │           Z^(0)     │           ×───────────×
  │           │         │           │           │
  @───────────┼─────────@───────────@^(-7/11)   │
  │           │         │           │           │
  │           │         ×───────────×           │
  │           │         │           │           │
  │           #2────────YXXY^0.123  │           │
  │           │         │           │           │
  │           Z^(0)     Z^(0)       │           │
  │           │         │           │           │
  │           │         #2──────────YXXY^-0.449 │
  │           │         │           │           │
  │           #2────────YXXY^0.301  Z^(0)       │
  │           │         │           │           │
  │           Z^(0)     Z^(0)       #2──────────YXXY
  │           │         │           │           │
  │           │         #2──────────YXXY^(1/3)  Rz(-π)
┌ │           │         │           │           │         ┐
│ │           #2────────YXXY        Rz(-0.0π)   │         │
│ @───────────┼─────────┼───────────┼───────────@^-0.318  │
└ │           │         │           │           │         ┘
  │           Rz(-0.0π) Rz(-0.0π)   │           Rz(π)
  @───────────┼─────────┼───────────@^-0.318    │
  │           │         │           │           │
  @───────────┼─────────@^-0.318    Rz(0.0π)    │
  │           │         │           │           │
  @───────────@^0.408   Rz(0.0π)    │           │
  │           │         │           │           │
  Rz(-0.318π) Rz(0.0π)  │           │           │
  │           │         │           │           │
  │           #2────────YXXY^-1     │           │
  │           │         │           │           │
  │           Z^(0)     #2──────────YXXY^(-1/3) │
  │           │         │           │           │
  │           │         Z^(0)       #2──────────YXXY^-1
  │           │         │           │           │
  │           #2────────YXXY^-0.301 Z^(0)       │
  │           │         │           │           │
  │           Z^(0)     #2──────────YXXY^0.449  │
  │           │         │           │           │
  │           │         Z^(0)       │           │
  │           │         │           │           │
  │           #2────────YXXY^-0.123 │           │
  │           │         │           │           │
  │           Z^(0)     │           │           │
  │           │         │           │           │
""", transpose=True)

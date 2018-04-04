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

from openfermion.utils._testing_utils import random_hermitian_matrix

from openfermioncirq import FSWAP, XXYY, LinearQubit

from openfermioncirq.swap_network import (
        second_order_trotter_step,
        swap_network)

def test_swap_network():
    n_qubits = 4
    qubits = [LinearQubit(i) for i in range(n_qubits)]

    circuit = cirq.Circuit.from_ops(
            swap_network(qubits, lambda i, j, q0, q1: XXYY(q0, q1)),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert (circuit.to_text_diagram(transpose=True).strip() == """
0    1    2    3
│    │    │    │
XXYY─XXYY XXYY─XXYY
│    │    │    │
×────×    ×────×
│    │    │    │
│    XXYY─XXYY │
│    │    │    │
│    ×────×    │
│    │    │    │
XXYY─XXYY XXYY─XXYY
│    │    │    │
×────×    ×────×
│    │    │    │
│    XXYY─XXYY │
│    │    │    │
│    ×────×    │
│    │    │    │
""".strip())

    circuit = cirq.Circuit.from_ops(
            swap_network(qubits, lambda i, j, q0, q1: XXYY(q0, q1),
                         fermionic=True, offset=True),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert (circuit.to_text_diagram(transpose=True).strip() == """
0    1    2    3
│    │    │    │
│    XXYY─XXYY │
│    │    │    │
│    ×ᶠ───×ᶠ   │
│    │    │    │
XXYY─XXYY XXYY─XXYY
│    │    │    │
×ᶠ───×ᶠ   ×ᶠ───×ᶠ
│    │    │    │
│    XXYY─XXYY │
│    │    │    │
│    ×ᶠ───×ᶠ   │
│    │    │    │
XXYY─XXYY XXYY─XXYY
│    │    │    │
×ᶠ───×ᶠ   ×ᶠ───×ᶠ
│    │    │    │
""".strip())

    n_qubits = 5
    qubits = [LinearQubit(i) for i in range(n_qubits)]

    circuit = cirq.Circuit.from_ops(
            swap_network(qubits, lambda i, j, q0, q1: (),
                         fermionic=True),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert (circuit.to_text_diagram(transpose=True).strip() == """
0  1  2  3  4
│  │  │  │  │
×ᶠ─×ᶠ ×ᶠ─×ᶠ │
│  │  │  │  │
│  ×ᶠ─×ᶠ ×ᶠ─×ᶠ
│  │  │  │  │
×ᶠ─×ᶠ ×ᶠ─×ᶠ │
│  │  │  │  │
│  ×ᶠ─×ᶠ ×ᶠ─×ᶠ
│  │  │  │  │
×ᶠ─×ᶠ ×ᶠ─×ᶠ │
│  │  │  │  │
""".strip())

    circuit = cirq.Circuit.from_ops(
            swap_network(qubits, lambda i, j, q0, q1: (),
                         offset=True),
            strategy=cirq.InsertStrategy.EARLIEST)
    assert (circuit.to_text_diagram(transpose=True).strip() == """
0 1 2 3 4
│ │ │ │ │
│ ×─× ×─×
│ │ │ │ │
×─× ×─× │
│ │ │ │ │
│ ×─× ×─×
│ │ │ │ │
×─× ×─× │
│ │ │ │ │
│ ×─× ×─×
│ │ │ │ │
""".strip())


def test_second_order_trotter_step():
    # Changing this value would break this test
    n_qubits = 5

    one_body = random_hermitian_matrix(n_qubits, real=True)
    two_body = random_hermitian_matrix(n_qubits, real=True)
    qubits = [LinearQubit(i) for i in range(n_qubits)]
    time_step = 1.

    circuit = cirq.Circuit.from_ops(
            second_order_trotter_step(qubits, one_body, two_body, time_step),
            strategy=cirq.InsertStrategy.EARLIEST)

    # Check gates
    operation = circuit.operation_at(qubits[2], 0)
    assert operation == cirq.Z(qubits[2]) ** (
            one_body[2, 2] * 0.5 * time_step / numpy.pi)

    operation = circuit.operation_at(qubits[2], 1)
    assert operation == XXYY(qubits[2], qubits[3]) ** (
            one_body[2, 3] * 0.5 * time_step / numpy.pi)

    operation = circuit.operation_at(qubits[2], 2)
    assert operation == cirq.CZ(qubits[2], qubits[3]) ** (
            two_body[2, 3] * 0.5 * time_step / numpy.pi)

    operation = circuit.operation_at(qubits[2], 3)
    assert operation == FSWAP(qubits[2], qubits[3])

    operation = circuit.operation_at(qubits[0], 16)
    assert operation == XXYY(qubits[0], qubits[1]) ** (
            one_body[4, 3] * 0.5 * time_step / numpy.pi)

    # Check circuit layout
    n_moments = len(circuit.moments)
    assert n_moments == 6 * n_qubits + 1

    moments = [circuit.next_moment_operating_on([qubits[0]], i)
               for i in range(n_moments)]
    assert moments == ([0] + [1, 2, 3] +
                       [7] * 3 + [7, 8, 9] +
                       [13] * 3 + [13, 14, 15, 16, 17, 18] +
                       [22] * 3 + [22, 23, 24] +
                       [28] * 3 + [28, 29, 30])

    moments = [circuit.next_moment_operating_on([qubits[1]], i)
               for i in range(n_moments)]
    assert moments == list(range(n_moments))

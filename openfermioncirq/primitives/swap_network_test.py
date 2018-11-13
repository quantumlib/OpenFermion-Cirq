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

import cirq

from openfermioncirq import XXYY, swap_network


def test_swap_network():
    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)

    circuit = cirq.Circuit.from_ops(
            swap_network(qubits, lambda i, j, q0, q1: XXYY(q0, q1)),
            strategy=cirq.InsertStrategy.EARLIEST)
    cirq.testing.assert_has_diagram(circuit, """
0: ───XXYY───×──────────────XXYY───×──────────────
      │      │              │      │
1: ───XXYY───×───XXYY───×───XXYY───×───XXYY───×───
                 │      │              │      │
2: ───XXYY───×───XXYY───×───XXYY───×───XXYY───×───
      │      │              │      │
3: ───XXYY───×──────────────XXYY───×──────────────
""")

    circuit = cirq.Circuit.from_ops(
            swap_network(qubits, lambda i, j, q0, q1: XXYY(q0, q1),
                         fermionic=True, offset=True),
            strategy=cirq.InsertStrategy.EARLIEST)
    cirq.testing.assert_has_diagram(circuit, """
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
""", transpose=True)

    n_qubits = 5
    qubits = cirq.LineQubit.range(n_qubits)

    circuit = cirq.Circuit.from_ops(
            swap_network(qubits, lambda i, j, q0, q1: (),
                         fermionic=True),
            strategy=cirq.InsertStrategy.EARLIEST)
    cirq.testing.assert_has_diagram(circuit, """
0: ───×ᶠ────────×ᶠ────────×ᶠ───
      │         │         │
1: ───×ᶠ───×ᶠ───×ᶠ───×ᶠ───×ᶠ───
           │         │
2: ───×ᶠ───×ᶠ───×ᶠ───×ᶠ───×ᶠ───
      │         │         │
3: ───×ᶠ───×ᶠ───×ᶠ───×ᶠ───×ᶠ───
           │         │
4: ────────×ᶠ────────×ᶠ────────
""")

    circuit = cirq.Circuit.from_ops(
            swap_network(qubits, lambda i, j, q0, q1: (),
                         offset=True),
            strategy=cirq.InsertStrategy.EARLIEST)
    cirq.testing.assert_has_diagram(circuit, """
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
""", transpose=True)


def test_reusable():
    ops = swap_network(cirq.LineQubit.range(5))
    assert list(ops) == list(ops)

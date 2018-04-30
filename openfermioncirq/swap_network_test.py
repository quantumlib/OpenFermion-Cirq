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
from cirq import LineQubit

from openfermioncirq import XXYY

from openfermioncirq.swap_network import swap_network

def test_swap_network():
    n_qubits = 4
    qubits = [LineQubit(i) for i in range(n_qubits)]

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
    qubits = [LineQubit(i) for i in range(n_qubits)]

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

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

"""The linear swap network."""

from typing import Callable, Sequence

import cirq

from openfermioncirq import FSWAP


def swap_network(qubits: Sequence[cirq.QubitId],
                 operation: Callable[
                     [int, int, cirq.QubitId, cirq.QubitId], cirq.OP_TREE]=
                     lambda p, q, p_qubit, q_qubit: (),
                 fermionic: bool=False,
                 offset: bool=False) -> cirq.OP_TREE:
    """Apply operations to pairs of qubits or modes using a swap network.

    This is used for applying operations between arbitrary pairs of qubits or
    fermionic modes using only nearest-neighbor interactions on a linear array
    of qubits. It works by reversing the order of qubits or modes with a
    sequence of swap gates and applying an operation when the relevant qubits
    or modes become adjacent. For fermionic modes, this assumes the
    Jordan-Wigner Transform.

    Args:
        qubits: The qubits sorted so that the j-th qubit in the Sequence
            represents the j-th qubit or fermionic mode
        operation: A call to this function takes the form
                operation(p, q, p_qubit, q_qubit)
            where p and q are indices reprenting either qubits or fermionic
            modes, and p_qubit and q_qubit are the qubits which represent them.
            It returns the gate that should be applied to these qubits.
        fermionic: If True, use fermionic swaps under the JWT (that is, swap
            fermionic modes instead of qubits). If False, use normal qubit
            swaps.
        offset: If True, then qubit 0 will participate in odd-numbered layers
            instead of even-numbered layers.
    """
    n_qubits = len(qubits)
    order = list(range(n_qubits))
    swap_gate = FSWAP if fermionic else cirq.SWAP

    for layer_num in range(n_qubits):
        lowest_active_qubit = (layer_num + offset) % 2
        active_pairs = ((i, i + 1)
                        for i in range(lowest_active_qubit, n_qubits - 1, 2))
        for i, j in active_pairs:
            p, q = order[i], order[j]
            yield operation(p, q, qubits[i], qubits[j])
            yield swap_gate(qubits[i], qubits[j])
            order[i], order[j] = q, p

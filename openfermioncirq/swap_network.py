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
from typing import Callable, Sequence

import numpy

import cirq
from cirq import SWAP

from openfermioncirq.gates import FSWAP, XXYY
from openfermioncirq.linear_qubit import LinearQubit


def swap_network(qubits: Sequence[LinearQubit],
                 operation: Callable[
                     [int, int, LinearQubit, LinearQubit], cirq.OP_TREE],
                 fermionic: bool=False,
                 offset: bool=False):
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
    swap_gate = FSWAP if fermionic else SWAP

    for layer_num in range(n_qubits):
        lowest_active_qubit = (layer_num + offset) % 2
        active_pairs = ((i, i + 1)
                        for i in range(lowest_active_qubit, n_qubits - 1, 2))

        for i, j in active_pairs:
            p, q = order[i], order[j]
            yield operation(p, q, qubits[i], qubits[j])
            yield swap_gate(qubits[i], qubits[j])
            order[i], order[j] = q, p


def second_order_trotter_step(qubits: Sequence[LinearQubit],
                              one_body: numpy.ndarray,
                              two_body: numpy.ndarray,
                              time: float):
    """Construct a second-order (first-order symmetric) Trotter step.

    This algorithm is described in arXiv:1711.04789. It assumes the
    Jordan-Wigner Transform.

    Args:
        qubits: The qubits on which to apply the Trotter step. They should
            be sorted so that the j-th qubit in the Sequence holds the
            occupation of the j-th fermionic mode.
        one_body: The matrix of coefficients T_{ij}. Currently only
            real coefficients are supported.
        two_body: The matrix of coefficients V_{ij}.
        time: The evolution time.
    """
    n_qubits = len(qubits)

    # Apply one-body potential
    for i in range(n_qubits):
        yield cirq.Z(qubits[i]) ** (one_body[i, i] * 0.5 * time / numpy.pi)

    # Define the operation to be performed on the modes
    def one_and_two_body_interaction(p, q, a, b):
        yield XXYY(a, b) ** (one_body[p, q] * 0.5 * time / numpy.pi)
        yield cirq.CZ(a, b) ** (two_body[p, q] * 0.5 * time / numpy.pi)

    # Perform the operations using fswap networks
    yield swap_network(qubits, one_and_two_body_interaction, fermionic=True)
    yield swap_network(list(reversed(qubits)), one_and_two_body_interaction,
                       fermionic=True, offset=True)

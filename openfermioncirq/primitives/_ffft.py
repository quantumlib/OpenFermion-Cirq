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

"""The fast fermionic Fourier transform."""

from typing import (List, Sequence)

import numpy as np

import cirq
import cirq.contrib.acquaintance.permutation
from openfermioncirq import FSWAP


class _F0Gate(cirq.TwoQubitMatrixGate):
    r"""Two-qubit gate that performs fermionic Fourier transform of size 2.

    Realizes unitary gate :math:`F_0` that transforms Fermionic creation
    operators :math:`a_0^\dagger` and :math:`a_1^\dagger` according to:

    .. math::
        F_0^\dagger a_0^\dagger F_0 =
            {1 \over \sqrt{2}} (a_0^\dagger + a_1^\dagger)

    .. math::
        F_0^\dagger a_1^\dagger F_0 =
            {1 \over \sqrt{2}} (a_0^\dagger - a_1^\dagger) \, .

    This gate assumes JWT representation of fermionic modes which are big-endian
    encoded on consecutive qubits:
    :math:`a_0^\dagger \lvert 0 \rangle = \lvert 10_2 \rangle` and
    :math:`a_1^\dagger \lvert 0 \rangle = \vert 01_2 \rangle`.

    Internally, this leads to expansion of :math:`F_0^\dagger`:

    .. math::
        \langle 0 \rvert F_0^\dagger \lvert 0 \rangle = 1

    .. math::
        \langle 01_2 \rvert F_0^\dagger \lvert 01_2 \rangle =
            -{1 \over \sqrt{2}}

    .. math::
        \langle 10_2 \rvert F_0^\dagger \lvert 10_2 \rangle =
        \langle 10_2 \rvert F_0^\dagger \lvert 01_2 \rangle =
        \langle 01_2 \rvert F_0^\dagger \lvert 10_2 \rangle = {1 \over \sqrt{2}}

    .. math::
        \langle 11_2 \rvert F_0^\dagger \lvert 11_2 \rangle = -1 \, .
    """

    def __init__(self):
        """Initializes :math:`F_0` gate."""
        cirq.TwoQubitMatrixGate.__init__(
            self,
            np.array([[1,          0,         0,  0],
                      [0, -2**(-0.5), 2**(-0.5),  0],
                      [0,  2**(-0.5), 2**(-0.5),  0],
                      [0,          0,         0, -1]]))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            symbols = 'F₀', 'F₀'
        else:
            symbols = 'F0', 'F0'
        return cirq.CircuitDiagramInfo(wire_symbols=symbols)


class _TwiddleGate(cirq.ZPowGate):
    r"""Gate that introduces arbitrary FFT twiddle factors.

    Realizes unitary gate :math:`\omega^{k\dagger}_n` that phases creation
    operator :math:`a^\dagger_x` according to:

    .. math::
        \omega^{k\dagger}_n a^\dagger_x \omega^k_n =
            e^{-2 \pi i {k \over n}} a^\dagger_x \, .

    Under JWT representation this is realized by appropriately rotated pauli Z
    gate acting on qubit x.
    """
    def __init__(self, k, n):
        """Initializes Twiddle gate.

        Args:
            k: Nominator appearing in the exponent.
            n: Denominator appearing in the exponent.
        """
        cirq.ZPowGate.__init__(self, exponent=-2 * k / n, global_shift=0)
        self.k = k
        self.n = n

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            symbols = 'ω^{}_{}'.format(self.k, self.n),
        else:
            symbols = 'w^{}_{}'.format(self.k, self.n),
        return cirq.CircuitDiagramInfo(wire_symbols=symbols)


F0 = _F0Gate()


def ffft(qubits: Sequence[cirq.QubitId]) -> cirq.OP_TREE:
    r"""TODO"""
    n = len(qubits)

    if n == 0:
        raise ValueError('Number of qubits is 0.')

    if n == 1:
        return []

    if n == 2:
        return F0(*qubits)

    if n % 2 != 0:
        raise ValueError('Number of qubits is not a power of 2.')

    ny = 2
    nx = n // ny
    permutation = [(i % ny) * nx + (i // ny) for i in range(n)]

    operations = []

    operations.append(_permute(qubits, permutation))

    for y in range(ny):
        operations.append(ffft(qubits[nx * y:nx * (y + 1)]))

    operations.append(_permute(qubits, _inverse(permutation)))

    for x in range(nx):
        for y in range(1, ny):
            operations.append(_TwiddleGate(x * y, n).on(qubits[ny * x + y]))
        operations.append(ffft(qubits[ny * x:ny * (x + 1)]))

    operations.append(_permute(qubits, permutation))

    return operations


def _inverse(permutation: List[int]) -> List[int]:
    """Calculates the inverse permutation function.

    Args:
        permutation: The list representation of a permutation function.

    Return:
        The inverse permutation function so that _compose(_inverse(permutation),
        permutation) is an identity.
    """
    inverse = [0] * len(permutation)
    for i in range(len(permutation)):
        inverse[permutation[i]] = i
    return inverse


def _permute(qubits: Sequence[cirq.QubitId],
             permutation: List[int]) -> cirq.OP_TREE:
    """
    Generates a circuit which reorders Fermionic modes.

    JWT representation of Fermionic modes is assumed. This is just a wrapper
    around cirq.contrib.acquaintance.permutation.LinearPermutationGate which
    internally uses bubble sort algorithm to generate permutation gate.

    Args:
        qubits: Sequence of qubits to reorder. Line connectivity is assumed.
        permutation: The permutation function represented as a list that
            reorders the initial qubits. Specifically, if k-th element of
            permutation is j, then k-th qubit should become the j-th qubit after
            applying the circuit to the initial state.

    Return:
        Gate that reorders the qubits accordingly.
    """
    return cirq.contrib.acquaintance.permutation.LinearPermutationGate(
        {i: permutation[i] for i in range(len(permutation))},
        swap_gate=FSWAP
    ).on(*qubits)

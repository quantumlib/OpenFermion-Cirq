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
import cirq.contrib.acquaintance
from openfermioncirq import FSWAP

class _F0Gate(cirq.TwoQubitMatrixGate):
    r"""Two-qubit gate that performs fermionic Fourier transform of size 2.

    More specifically, realizes unitary gate :math:`F_0` that transforms
    Fermionic creation operators :math:`a_0^\dagger` and :math:`a_1^\dagger`
    according to:

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


def _shift(permutation: List[int], shift: int) -> List[int]:
    """Add a constant to every element of permutation.

    Args:
        permutation: The list representation of a permutation function.
        shift: Constant offset.

    Return:
        Permutation function with every element shifted by a constant. For
        non-zero shift this is technically not a permutation anymore, but might
        be used as a part of higher dimensional permutation.
    """
    return [p + shift for p in permutation]


def _inverse(permutation: List[int]) -> List[int]:
    """Calculates the inverse permutation function.

    Args:
        permutation: The list representation of a permutation function.

    Return:
        The inverse permutation function so that _compose(_inverse(permutation),
        permutation) is an identity.
    """
    inverse = [0]*len(permutation)
    for i in range(len(permutation)):
        inverse[permutation[i]] = i
    return inverse


def _compose(outer: List[int], inner: List[int]) -> List[int]:
    """Creates the composition of the permutation functions.

    Args:
        outer: The outer permutation function represented as a list. Length must
        match the length of the inner permutation function.
        inner: The inner permutation function represented as a list. Length must
        match the length of the outer permutation function.

    Return:
         Permutation function which is a composition of outer and inner. The
         returned permutation function is outer[inner[x]] for each permuted
         index x in range.
    """
    return [outer[i] for i in inner]


def _permute(qubits: Sequence[cirq.QubitId],
             permutation: List[int]) -> cirq.OP_TREE:
    """
    Generates a circuit which reorders qubits using bubble sort algorithm.

    Args:
        qubits: Sequence of qubits to reorder. It is assumed they have line
            connectivity.
        permutation: The permutation function represented as a list that
            reorders the initial qubits. Specifically, if k-th element of
            permutation is j, then k-th qubit should become the j-th qubit after
            applying the circuit to the initial state.
            This is so called active representation of permutation which is a
            function that performs rearrangement, not a rearrangement itself.
            This representation behaves well under composition.

    Return:
        Gate that reorders the qubits accordingly.
    """
    yield cirq.contrib.acquaintance.LinearPermutationGate(
        {i: permutation[i] for i in range(len(permutation))},
        swap_gate=FSWAP
    ).on(*qubits)

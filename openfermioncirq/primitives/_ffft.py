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

import cirq


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


def _permute_bubble(qubits: Sequence[cirq.QubitId],
                    permutation: List[int],
                    swap: cirq.TwoQubitGate) -> cirq.OP_TREE:
    """
    Generates a circuit which reorders qubits using bubble sort algorithm.

    The algorithm works on a sequence of qubits and at each step applies the
    swap gate between two adjacent qubits.

    Iterates over permutation multiple times and swaps neighbouring elements if
    they're inverted, until no more swap is necessary.

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
        swap: Swap gate to be applied when performing a single swap.
            To realize qubit swap use cirq.SWAP gate.
            To realize fermionic swap with assumption that qubits are
            represented under JWT transform, use openfermioncirq.FSWAP gate.

    Return:
        Generator of the operations that represent the sorting circuit. The
        algorithm generates a one dimensional sequence of swap gates on
        particular qubits.
    """
    per = list(permutation)
    swapped = True
    while swapped:
        swapped = False
        j = 1
        while j < len(per):
            if per[j-1] > per[j]:
                yield swap(qubits[j - 1], qubits[j])
                per[j-1], per[j] = per[j], per[j-1]
                swapped = True
                j += 1
            j += 1

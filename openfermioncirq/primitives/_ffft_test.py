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

import numpy as np
import pytest

import cirq
from openfermioncirq.primitives._ffft import (
    _compose,
    _inverse,
    _permute,
    _shift
)

from cirq import LineQubit


def _state_from_modes(n, amplitude, modes):
    """Prepares a state from a list of desired modes.

    Prepares a state to be one of the basis vectors of an n-dimensional qubits
    Hilbert space. The basis state has qubits from the list modes set to 1, and
    all other qubits set to 0.

    Args:
         n: State length, number of qubits used.
         amplitude: State amplitude. Absolute value must be equal to 1.
         modes: List of modes number which should appear in the resulting state.

    Return:
        State vector that represents n-dimensional Hilbert space base state,
        with listed modes prepared in state 1. State vector is big-endian
        encoded to match Cirq conventions.
    """
    state = np.zeros(1 << n, dtype=complex)
    state[sum(1 << (n - 1 - m) for m in modes)] = amplitude
    return state


def test_shift():
    assert _shift([], 1) == []
    assert _shift([0, 1, 2], 0) == [0, 1, 2]
    assert _shift([2, 0, 1], 2) == [4, 2, 3]
    assert _shift([2, 0, 1], -2) == [0, -2, -1]


def test_inverse():
    assert _inverse([0, 1, 2]) == [0, 1, 2]
    assert _inverse([1, 2, 0]) == [2, 0, 1]
    assert _inverse([2, 0, 1]) == [1, 2, 0]
    assert _inverse([3, 2, 1, 0]) == [3, 2, 1, 0]


def test_compose():
    assert _compose([0, 1, 2], [1, 2, 0]) == [1, 2, 0]
    assert _compose([1, 2, 0], [0, 1, 2]) == [1, 2, 0]
    assert _compose([1, 2, 0], [1, 2, 0]) == [2, 0, 1]
    assert _compose([2, 0, 1], [1, 2, 0]) == [0, 1, 2]
    assert _compose([1, 2, 0], [2, 0, 1]) == [0, 1, 2]
    assert _compose([0, 2, 1], [1, 2, 0]) == [2, 1, 0]
    assert _compose([1, 2, 0], [0, 2, 1]) == [1, 0, 2]


@pytest.mark.parametrize(
    'permutation',
    [[],
     [0],
     [0, 1],
     [1, 0],
     [2, 0, 1],
     [2, 3, 0, 1],
     [5, 0, 2, 4, 3, 1]]
)
def test_compose_inverse_identity(permutation):
    identity = list(range(len(permutation)))
    assert _compose(_inverse(permutation), permutation) == identity
    assert _compose(permutation, _inverse(permutation)) == identity


@pytest.mark.parametrize(
        'permutation, initial, expected',
        [([0, 1], (1, []), (1, [])),
         ([0, 1], (1, [1]), (1, [1])),
         ([1, 0], (1, [1]), (1, [0])),
         ([1, 2, 0], (1, [0]), (1, [1])),
         ([1, 2, 0], (1, [1]), (1, [2])),
         ([1, 2, 0], (1, [2]), (1, [0])),
         ([1, 0], (1, [0, 1]), (-1, [0, 1])),
         ([2, 1, 0], (1, [0, 2]), (-1, [0, 2])),
         ([2, 1, 0], (1j, [0, 2]), (-1j, [0, 2])),
         ([1, 0, 2], (1, [0, 2]), (1, [1, 2])),
         ([0, 1, 2, 3, 4], (1, [0]), (1, [0])),
         ([1, 0, 2, 3, 4], (1, [0]), (1, [1])),
         ([2, 1, 0, 3, 4], (1, [0]), (1, [2])),
         ([3, 1, 2, 0, 4], (1, [0]), (1, [3])),
         ([4, 1, 2, 3, 0], (1, [0]), (1, [4])),
         ([3, 1, 2, 0, 4], (1, [0, 4]), (1, [3, 4])),
         ([3, 1, 2, 0, 4], (1, [0, 3]), (-1, [0, 3])),
         ([3, 1, 2, 0, 4], (1, [0, 2]), (-1, [2, 3])),
         ([0, 2, 4, 6, 1, 3, 5, 7], (1, [3, 4]), (-1, [1, 6])),
         ([0, 2, 4, 6, 1, 3, 5, 7], (1, [2, 3, 4]), (1, [1, 4, 6])),
         ([7, 6, 5, 4, 3, 2, 1, 0], (1, [2, 3, 4]), (-1, [3, 4, 5]))]
)
def test_permute(permutation, initial, expected):
    n = len(permutation)
    initial_state = _state_from_modes(n, *initial)
    expected_state = _state_from_modes(n, *expected)
    qubits = LineQubit.range(n)

    ops = _permute(qubits, permutation)
    circuit = cirq.Circuit.from_ops(ops)

    state = circuit.apply_unitary_effect_to_state(
        initial_state,
        qubits_that_should_be_present=qubits
    )

    assert np.allclose(state, expected_state, rtol=0.0)

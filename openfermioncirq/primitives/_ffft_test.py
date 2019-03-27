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
from cirq import LineQubit
from openfermioncirq import (bogoliubov_transform, ffft)
from openfermioncirq.primitives._ffft import (
    _F0Gate,
    _TwiddleGate,
    _inverse,
    _permute,
)


def _state_from_amplitudes(amplitudes):
    """Prepares state in a superposition of Fermionic modes.

    Args:
        amplitudes: List of amplitudes to be assigned for a state representing
            a Fermionic mode.

    Return:
        State vector which is a superposition of single fermionic modes under
        JWT representation, each with appropriate amplitude assigned.
    """
    n = len(amplitudes)
    state = np.zeros(1 << n, dtype=complex)
    for m in range(len(amplitudes)):
        state[1 << (n - 1 - m)] = amplitudes[m]
    return state


def _fft_amplitudes(amplitudes):
    """Fermionic Fourier transform of Fermionic modes.

    Args:
        amplitudes: List of amplitudes for each Fermionic mode.

    Return:
        List representing a new, Fourier transformed amplitudes of the input
        modes amplitudes.
    """
    def fft(k, n):
        unit = np.exp(-2j * np.pi * k / n)
        return sum(unit**j * amplitudes[j] for j in range(n)) / np.sqrt(n)
    n = len(amplitudes)
    return [fft(k, n) for k in range(n)]


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


@pytest.mark.parametrize(
        'amplitudes',
        [[1, 0],
         [1j, 0],
         [0, 1],
         [0, -1j],
         [np.sqrt(2), np.sqrt(2)]]
)
def test_F0Gate_transform(amplitudes):
    qubits = LineQubit.range(2)
    initial_state = _state_from_amplitudes(amplitudes)
    expected_state = _state_from_amplitudes(_fft_amplitudes(amplitudes))

    circuit = cirq.Circuit.from_ops(_F0Gate().on(*qubits))
    state = circuit.apply_unitary_effect_to_state(initial_state)

    assert np.allclose(state, expected_state, rtol=0.0)


def test_F0Gate_text_unicode_diagram():
    qubits = LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(_F0Gate().on(*qubits))

    assert circuit.to_text_diagram().strip() == """
0: ───F₀───
      │
1: ───F₀───
    """.strip()


def test_F0Gate_text_diagram():
    qubits = LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(_F0Gate().on(*qubits))

    assert circuit.to_text_diagram(use_unicode_characters=False).strip() == """
0: ---F0---
      |
1: ---F0---
    """.strip()


@pytest.mark.parametrize(
        'k, n, qubit, initial, expected',
        [(0, 2, 0, [1, 0], [1, 0]),
         (2, 8, 0, [1, 0], [np.exp(-2 * np.pi * 1j * 2 / 8), 0]),
         (4, 8, 1, [0, 1], [0, np.exp(-2 * np.pi * 1j * 4 / 8)]),
         (3, 5, 0, [1, 1], [np.exp(-2 * np.pi * 1j * 3 / 5), 1]),]
)
def test_TwiddleGate_transform(k, n, qubit, initial, expected):
    qubits = LineQubit.range(2)
    initial_state = _state_from_amplitudes(initial)
    expected_state = _state_from_amplitudes(expected)

    circuit = cirq.Circuit.from_ops(_TwiddleGate(k, n).on(qubits[qubit]))
    state = circuit.apply_unitary_effect_to_state(
        initial_state,
        qubits_that_should_be_present=qubits
    )

    assert np.allclose(state, expected_state, rtol=0.0)


def test_TwiddleGate_text_unicode_diagram():
    qubit = LineQubit.range(1)
    circuit = cirq.Circuit.from_ops(_TwiddleGate(2, 8).on(*qubit))

    assert circuit.to_text_diagram().strip() == """
0: ───ω^2_8───
    """.strip()


def test_TwiddleGate_text_diagram():
    qubit = LineQubit.range(1)
    circuit = cirq.Circuit.from_ops(_TwiddleGate(2, 8).on(*qubit))

    assert circuit.to_text_diagram(use_unicode_characters=False).strip() == """
0: ---w^2_8---
    """.strip()


@pytest.mark.parametrize(
        'amplitudes',
        [[1],
         [1, 0],
         [0, 1],
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, -1j/np.sqrt(2), 0, 0, 1/np.sqrt(2), 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         ])
def test_ffft_single_mode(amplitudes):
    initial_state = _state_from_amplitudes(amplitudes)
    expected_state = _state_from_amplitudes(_fft_amplitudes(amplitudes))
    qubits = LineQubit.range(len(amplitudes))

    circuit = cirq.Circuit.from_ops(
        ffft(qubits), strategy=cirq.InsertStrategy.EARLIEST)
    state = circuit.apply_unitary_effect_to_state(
        initial_state, qubits_that_should_be_present=qubits)

    assert np.allclose(state, expected_state, rtol=0.0)


def test_ffft_text_diagram():
    qubits = LineQubit.range(8)

    circuit = cirq.Circuit.from_ops(
        ffft(qubits), strategy=cirq.InsertStrategy.EARLIEST)

    assert circuit.to_text_diagram(transpose=True) == """
0   1     2   3     4   5     6   7
│   │     │   │     │   │     │   │
0↦0─1↦4───2↦1─3↦5───4↦2─5↦6───6↦3─7↦7
│   │     │   │     │   │     │   │
0↦0─1↦2───2↦1─3↦3   0↦0─1↦2───2↦1─3↦3
│   │     │   │     │   │     │   │
F₀──F₀    F₀──F₀    F₀──F₀    F₀──F₀
│   │     │   │     │   │     │   │
0↦0─1↦2───2↦1─3↦3   0↦0─1↦2───2↦1─3↦3
│   │     │   │     │   │     │   │
│   ω^0_4 │   ω^1_4 │   ω^0_4 │   ω^1_4
│   │     │   │     │   │     │   │
F₀──F₀    F₀──F₀    F₀──F₀    F₀──F₀
│   │     │   │     │   │     │   │
0↦0─1↦2───2↦1─3↦3   0↦0─1↦2───2↦1─3↦3
│   │     │   │     │   │     │   │
0↦0─1↦2───2↦4─3↦6───4↦1─5↦3───6↦5─7↦7
│   │     │   │     │   │     │   │
│   ω^0_8 │   ω^1_8 │   ω^2_8 │   ω^3_8
│   │     │   │     │   │     │   │
F₀──F₀    F₀──F₀    F₀──F₀    F₀──F₀
│   │     │   │     │   │     │   │
0↦0─1↦4───2↦1─3↦5───4↦2─5↦6───6↦3─7↦7
│   │     │   │     │   │     │   │
    """.strip()


def test_ffft_fails_without_qubits():
    with pytest.raises(ValueError):
        ffft([])


def test_ffft_fails_for_odd_size():
    with pytest.raises(ValueError):
        ffft(LineQubit.range(3))


@pytest.mark.parametrize('size', [1, 2, 4, 8])
def test_ffft_equal_to_bogoliubov(size):

    def fourier_transform_matrix():
        root_of_unity = np.exp(-2j * np.pi / size)
        return np.array([[root_of_unity ** (j * k) for k in range(size)]
                        for j in range(size)]) / np.sqrt(size)

    qubits = LineQubit.range(size)

    ffft_circuit = cirq.Circuit.from_ops(
        ffft(qubits), strategy=cirq.InsertStrategy.EARLIEST)
    ffft_matrix = ffft_circuit.to_unitary_matrix(
        qubits_that_should_be_present=qubits)

    bogoliubov_circuit = cirq.Circuit.from_ops(
        bogoliubov_transform(qubits, fourier_transform_matrix()),
        strategy=cirq.InsertStrategy.EARLIEST)
    bogoliubov_matrix = bogoliubov_circuit.to_unitary_matrix(
        qubits_that_should_be_present=qubits)

    cirq.testing.assert_allclose_up_to_global_phase(
        ffft_matrix, bogoliubov_matrix, atol=1e-8
    )


def test_inverse():
    assert _inverse([0, 1, 2]) == [0, 1, 2]
    assert _inverse([1, 2, 0]) == [2, 0, 1]
    assert _inverse([2, 0, 1]) == [1, 2, 0]
    assert _inverse([3, 2, 1, 0]) == [3, 2, 1, 0]


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

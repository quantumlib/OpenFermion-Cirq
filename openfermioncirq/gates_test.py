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
import pytest

import numpy
from scipy.linalg import expm, kron

import cirq
from cirq import LineQubit
from cirq.testing import EqualsTester

from openfermioncirq.gates import (CCZ, CXXYY, CYXXY, FSWAP, XXYY, XXYYGate,
                                   YXXY, YXXYGate)


def test_fswap_interchangeable():
    a, b = LineQubit(0), LineQubit(1)
    assert FSWAP(a, b) == FSWAP(b, a)


def test_fswap_on_simulator():
    simulator = cirq.google.Simulator()
    a, b = cirq.google.XmonQubit(0, 0), cirq.google.XmonQubit(1, 0)
    circuit = cirq.Circuit.from_ops(FSWAP(a, b))

    initial_state = (numpy.array([1, 1, 0, 0], dtype=numpy.complex64) /
                     numpy.sqrt(2))
    result = simulator.run(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_states[0],
            numpy.array([1, 0, 1, 0]) / numpy.sqrt(2),
            atol=5e-6)

    initial_state = (numpy.array([0, 1, 0, 1], dtype=numpy.complex64) /
                     numpy.sqrt(2))
    result = simulator.run(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_states[0],
            numpy.array([0, 0, 1, -1]) / numpy.sqrt(2),
            atol=1e-7)


def test_xx_yy_init():
    assert XXYYGate(half_turns=0.5).half_turns == 0.5
    assert XXYYGate(half_turns=5).half_turns == 1


def test_xx_yy_eq():
    eq = EqualsTester()
    eq.add_equality_group(XXYYGate(half_turns=3.5),
                          XXYYGate(half_turns=-0.5))
    eq.make_equality_pair(lambda: XXYYGate(half_turns=0))
    eq.make_equality_pair(lambda: XXYYGate(half_turns=0.5))


def test_xx_yy_interchangeable():
    a, b = LineQubit(0), LineQubit(1)
    assert XXYY(a, b) == XXYY(b, a)


def test_xx_yy_extrapolate():
    assert XXYYGate(
        half_turns=1).extrapolate_effect(0.5) == XXYYGate(half_turns=0.5)


def test_xx_yy__matrix():
    numpy.testing.assert_allclose(XXYYGate(half_turns=2).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-7)

    numpy.testing.assert_allclose(XXYYGate(half_turns=1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, -1j, 0],
                                               [0, -1j, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-7)

    numpy.testing.assert_allclose(XXYYGate(half_turns=0).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-7)

    numpy.testing.assert_allclose(XXYYGate(half_turns=-1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, 1j, 0],
                                               [0, 1j, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-7)

    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    XX = kron(X, X)
    YY = kron(Y, Y)
    numpy.testing.assert_allclose(XXYYGate(half_turns=0.25).matrix(),
                                  expm(-1j * numpy.pi * 0.25 * (XX + YY) / 4))


@pytest.mark.parametrize(
        'half_turns, initial_state, correct_state, atol', [
            (1.0, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, -1j, -1j, 0]) / numpy.sqrt(2), 5e-6),
            (0.5, numpy.array([1, 1, 0, 0]) / numpy.sqrt(2),
                  numpy.array([1 / numpy.sqrt(2), 0.5, -0.5j, 0]), 1e-7),
            (-0.5, numpy.array([1, 1, 0, 0]) / numpy.sqrt(2),
                   numpy.array([1 / numpy.sqrt(2), 0.5, 0.5j, 0]), 1e-7),
])
def test_xxyy_on_simulator(half_turns, initial_state, correct_state, atol):

    simulator = cirq.google.Simulator()
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(XXYY(a, b)**half_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.run(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_states[0], correct_state, atol=atol)


def test_yx_xy_init():
    assert YXXYGate(half_turns=0.5).half_turns == 0.5
    assert YXXYGate(half_turns=5).half_turns == 1


def test_yx_xy_eq():
    eq = EqualsTester()
    eq.add_equality_group(YXXYGate(half_turns=3.5),
                          YXXYGate(half_turns=-0.5))
    eq.make_equality_pair(lambda: YXXYGate(half_turns=0))
    eq.make_equality_pair(lambda: YXXYGate(half_turns=0.5))


def test_yx_xy_extrapolate():
    assert YXXYGate(
        half_turns=1).extrapolate_effect(0.5) == YXXYGate(half_turns=0.5)


def test_yx_xy__matrix():
    numpy.testing.assert_allclose(YXXYGate(half_turns=2).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-7)

    numpy.testing.assert_allclose(YXXYGate(half_turns=1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-7)

    numpy.testing.assert_allclose(YXXYGate(half_turns=0).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-7)

    numpy.testing.assert_allclose(YXXYGate(half_turns=-1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-7)

    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    YX = kron(Y, X)
    XY = kron(X, Y)
    numpy.testing.assert_allclose(YXXYGate(half_turns=0.25).matrix(),
                                  expm(-1j * numpy.pi * 0.25 * (YX - XY) / 4))


@pytest.mark.parametrize(
        'half_turns, initial_state, correct_state, atol', [
            (1.0, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 1, -1, 0]) / numpy.sqrt(2), 1e-7),
            (0.5, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 0, 1, 0]), 1e-7),
            (-0.5, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                   numpy.array([0, 1, 0, 0]), 1e-7)
])
def test_yxxy_on_simulator(half_turns, initial_state, correct_state, atol):

    simulator = cirq.google.Simulator()
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(YXXY(a, b)**half_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.run(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_states[0], correct_state, atol=atol)


@pytest.mark.parametrize(
        'half_turns, initial_state, correct_state, atol', [
            (0.5, numpy.array([1, 0, 0, 0, 0, 0, 0, 1]) / numpy.sqrt(2),
                  numpy.array([1, 0, 0, 0, 0, 0, 0, 1j]) / numpy.sqrt(2), 5e-6),
            (1.0, numpy.array([0, 1, 0, 0, 0, 0, 0, 1]) / numpy.sqrt(2),
                  numpy.array([0, 1, 0, 0, 0, 0, 0, -1]) / numpy.sqrt(2), 5e-6),
            (0.5, numpy.array([0, 0, 1, 0, 0, 0, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 0, 1, 0, 0, 0, 1, 0]) / numpy.sqrt(2), 1e-7),
            (0.25, numpy.array([0, 0, 0, 1, 0, 1, 0, 0]) / numpy.sqrt(2),
                   numpy.array([0, 0, 0, 1, 0, 1, 0, 0]) / numpy.sqrt(2), 5e-6),
            (-0.5, numpy.array([0, 0, 0, 0, 1, 0, 0, 1j]) / numpy.sqrt(2),
                   numpy.array([0, 0, 0, 0, 1, 0, 0, 1]) / numpy.sqrt(2), 1e-7)
])
def test_ccz_on_simulator(half_turns, initial_state, correct_state, atol):

    simulator = cirq.google.Simulator()
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(CCZ(a, b, c)**half_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.run(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_states[0], correct_state, atol=atol)


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cxxyy_decompose(half_turns):

    gate = CXXYY**half_turns
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, gate.matrix(), atol=1e-7)


@pytest.mark.parametrize(
        'half_turns, initial_state, correct_state, atol', [
            (1.0, numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 0, 0, 0, 0, -1j, -1j, 0]) / numpy.sqrt(2),
                  5e-6),
            (0.5, numpy.array([0, 0, 0, 0, 1, 1, 0, 0]) / numpy.sqrt(2),
                  numpy.array([0, 0, 0, 0, 1 / numpy.sqrt(2), 0.5, -0.5j, 0]),
                  5e-6),
            (-0.5, numpy.array([0, 0, 0, 0, 1, 1, 0, 0]) / numpy.sqrt(2),
                   numpy.array([0, 0, 0, 0, 1 / numpy.sqrt(2), 0.5, 0.5j, 0]),
                   5e-6),
            (1.0, numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
                  numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, -0.5j, -0.5j, 0]),
                  5e-6),
            (1.0, numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                  numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                  5e-6),
            (0.5, numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                  numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                  5e-6),
            (-0.5, numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
                   numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
                   5e-6)
])
def test_cxxyy_on_simulator(half_turns, initial_state, correct_state, atol):

    simulator = cirq.google.Simulator()
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(CXXYY(a, b, c)**half_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.run(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_states[0], correct_state, atol=atol)


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cyxxy_decompose(half_turns):

    gate = CYXXY**half_turns
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, gate.matrix(), atol=1e-7)


@pytest.mark.parametrize(
        'half_turns, initial_state, correct_state, atol', [
            (1.0, numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 0, 0, 0, 0, 1, -1, 0]) / numpy.sqrt(2),
                  5e-6),
            (0.5, numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 0, 0, 0, 0, 0, 1, 0]),
                  5e-6),
            (-0.5, numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                   numpy.array([0, 0, 0, 0, 0, 1, 0, 0]),
                   5e-6),
            (-0.5, numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
                   numpy.array([1, 0, 0, 0, 0, 1, 0, 0]) / numpy.sqrt(2),
                   5e-6),
            (1.0, numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                  numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                  5e-6),
            (0.5, numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                  numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                  5e-6),
            (-0.5, numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
                   numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
                   5e-6)
])
def test_cyxxy_on_simulator(half_turns, initial_state, correct_state, atol):

    simulator = cirq.google.Simulator()
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(CYXXY(a, b, c)**half_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.run(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_states[0], correct_state, atol=atol)

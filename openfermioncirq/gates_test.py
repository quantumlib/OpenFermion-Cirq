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

from openfermioncirq.gates import (
        CCZ, CXXYY, CYXXY, ControlledXXYYGate, ControlledYXXYGate, Rot111Gate,
        FSWAP, XXYY, XXYYGate, YXXY, YXXYGate)


def test_fswap_interchangeable():
    a, b = LineQubit(0), LineQubit(1)
    assert FSWAP(a, b) == FSWAP(b, a)


def test_fswap_on_simulator():
    simulator = cirq.google.XmonSimulator()
    a, b = cirq.google.XmonQubit(0, 0), cirq.google.XmonQubit(1, 0)
    circuit = cirq.Circuit.from_ops(FSWAP(a, b))

    initial_state = (numpy.array([1, 1, 0, 0], dtype=numpy.complex64) /
                     numpy.sqrt(2))
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_state,
            numpy.array([1, 0, 1, 0]) / numpy.sqrt(2),
            atol=1e-7)

    initial_state = (numpy.array([0, 1, 0, 1], dtype=numpy.complex64) /
                     numpy.sqrt(2))
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_state,
            numpy.array([0, 0, 1, -1]) / numpy.sqrt(2),
            atol=1e-7)


def test_xxyy_init():
    assert XXYYGate(quarter_turns=0.5).quarter_turns == 0.5
    assert XXYYGate(quarter_turns=5).quarter_turns == 1


def test_xxyy_eq():
    eq = EqualsTester()
    eq.add_equality_group(XXYYGate(quarter_turns=3.5),
                          XXYYGate(quarter_turns=-0.5))
    eq.make_equality_pair(lambda: XXYYGate(quarter_turns=0))
    eq.make_equality_pair(lambda: XXYYGate(quarter_turns=0.5))


def test_xxyy_interchangeable():
    a, b = LineQubit(0), LineQubit(1)
    assert XXYY(a, b) == XXYY(b, a)


def test_xxyy_extrapolate():
    assert XXYYGate(
        quarter_turns=1).extrapolate_effect(0.5) == XXYYGate(quarter_turns=0.5)


def test_xxyy_repr():
    assert repr(XXYYGate(quarter_turns=1)) == 'XXYY'
    assert repr(XXYYGate(quarter_turns=0.5)) == 'XXYY**0.5'


@pytest.mark.parametrize('quarter_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_xxyy_decompose(quarter_turns):

    gate = XXYY**quarter_turns
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, gate.matrix(), atol=1e-8)


def test_xxyy__matrix():
    numpy.testing.assert_allclose(XXYYGate(quarter_turns=2).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(XXYYGate(quarter_turns=1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, -1j, 0],
                                               [0, -1j, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(XXYYGate(quarter_turns=0).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(XXYYGate(quarter_turns=-1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, 1j, 0],
                                               [0, 1j, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    XX = kron(X, X)
    YY = kron(Y, Y)
    numpy.testing.assert_allclose(XXYYGate(quarter_turns=0.25).matrix(),
                                  expm(-1j * numpy.pi * 0.25 * (XX + YY) / 4))


@pytest.mark.parametrize(
        'quarter_turns, initial_state, correct_state, atol', [
            (1.0, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, -1j, -1j, 0]) / numpy.sqrt(2), 1e-7),
            (0.5, numpy.array([1, 1, 0, 0]) / numpy.sqrt(2),
                  numpy.array([1 / numpy.sqrt(2), 0.5, -0.5j, 0]), 1e-7),
            (-0.5, numpy.array([1, 1, 0, 0]) / numpy.sqrt(2),
                   numpy.array([1 / numpy.sqrt(2), 0.5, 0.5j, 0]), 1e-7),
])
def test_xxyy_on_simulator(quarter_turns, initial_state, correct_state, atol):

    simulator = cirq.google.XmonSimulator()
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(XXYY(a, b)**quarter_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_state, correct_state, atol=atol)


def test_yxxy_init():
    assert YXXYGate(quarter_turns=0.5).quarter_turns == 0.5
    assert YXXYGate(quarter_turns=5).quarter_turns == 1


def test_yxxy_eq():
    eq = EqualsTester()
    eq.add_equality_group(YXXYGate(quarter_turns=3.5),
                          YXXYGate(quarter_turns=-0.5))
    eq.make_equality_pair(lambda: YXXYGate(quarter_turns=0))
    eq.make_equality_pair(lambda: YXXYGate(quarter_turns=0.5))


def test_yxxy_extrapolate():
    assert YXXYGate(
        quarter_turns=1).extrapolate_effect(0.5) == YXXYGate(quarter_turns=0.5)


def test_yxxy_repr():
    assert repr(YXXYGate(quarter_turns=1)) == 'YXXY'
    assert repr(YXXYGate(quarter_turns=0.5)) == 'YXXY**0.5'


@pytest.mark.parametrize('quarter_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_yxxy_decompose(quarter_turns):

    gate = YXXY**quarter_turns
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, gate.matrix(), atol=1e-8)


def test_yxxy__matrix():
    numpy.testing.assert_allclose(YXXYGate(quarter_turns=2).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(YXXYGate(quarter_turns=1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(YXXYGate(quarter_turns=0).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(YXXYGate(quarter_turns=-1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    YX = kron(Y, X)
    XY = kron(X, Y)
    numpy.testing.assert_allclose(YXXYGate(quarter_turns=0.25).matrix(),
                                  expm(-1j * numpy.pi * 0.25 * (YX - XY) / 4))


@pytest.mark.parametrize(
        'quarter_turns, initial_state, correct_state, atol', [
            (1.0, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 1, -1, 0]) / numpy.sqrt(2), 1e-8),
            (0.5, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 0, 1, 0]), 1e-7),
            (-0.5, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                   numpy.array([0, 1, 0, 0]), 1e-7)
])
def test_yxxy_on_simulator(quarter_turns, initial_state, correct_state, atol):

    simulator = cirq.google.XmonSimulator()
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(YXXY(a, b)**quarter_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_state, correct_state, atol=atol)


def test_ccz_repr():
    assert repr(Rot111Gate(half_turns=1)) == 'CCZ'
    assert repr(Rot111Gate(half_turns=0.5)) == 'CCZ**0.5'


@pytest.mark.parametrize(
        'half_turns, initial_state, correct_state, atol', [
            (0.5, numpy.array([1, 0, 0, 0, 0, 0, 0, 1]) / numpy.sqrt(2),
                  numpy.array([1, 0, 0, 0, 0, 0, 0, 1j]) / numpy.sqrt(2), 1e-7),
            (1.0, numpy.array([0, 1, 0, 0, 0, 0, 0, 1]) / numpy.sqrt(2),
                  numpy.array([0, 1, 0, 0, 0, 0, 0, -1]) / numpy.sqrt(2), 1e-7),
            (0.5, numpy.array([0, 0, 1, 0, 0, 0, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 0, 1, 0, 0, 0, 1, 0]) / numpy.sqrt(2), 1e-7),
            (0.25, numpy.array([0, 0, 0, 1, 0, 1, 0, 0]) / numpy.sqrt(2),
                   numpy.array([0, 0, 0, 1, 0, 1, 0, 0]) / numpy.sqrt(2), 1e-7),
            (-0.5, numpy.array([0, 0, 0, 0, 1, 0, 0, 1j]) / numpy.sqrt(2),
                   numpy.array([0, 0, 0, 0, 1, 0, 0, 1]) / numpy.sqrt(2), 1e-7)
])
def test_ccz_on_simulator(half_turns, initial_state, correct_state, atol):

    simulator = cirq.google.XmonSimulator()
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(CCZ(a, b, c)**half_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_state, correct_state, atol=atol)


@pytest.mark.parametrize('quarter_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cxxyy_decompose(quarter_turns):

    gate = CXXYY**quarter_turns
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, gate.matrix(), atol=1e-7)


def test_cxxyy_repr():
    assert repr(ControlledXXYYGate(quarter_turns=1)) == 'CXXYY'
    assert repr(ControlledXXYYGate(quarter_turns=0.5)) == 'CXXYY**0.5'


@pytest.mark.parametrize(
        'quarter_turns, initial_state, correct_state, atol', [
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
def test_cxxyy_on_simulator(quarter_turns, initial_state, correct_state, atol):

    simulator = cirq.google.XmonSimulator()
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(CXXYY(a, b, c)**quarter_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_state, correct_state, atol=atol)


@pytest.mark.parametrize('quarter_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cyxxy_decompose(quarter_turns):

    gate = CYXXY**quarter_turns
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, gate.matrix(), atol=1e-7)


def test_cyxxy_repr():
    assert repr(ControlledYXXYGate(quarter_turns=1)) == 'CYXXY'
    assert repr(ControlledYXXYGate(quarter_turns=0.5)) == 'CYXXY**0.5'


@pytest.mark.parametrize(
        'quarter_turns, initial_state, correct_state, atol', [
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
def test_cyxxy_on_simulator(quarter_turns, initial_state, correct_state, atol):

    simulator = cirq.google.XmonSimulator()
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(CYXXY(a, b, c)**quarter_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_state, correct_state, atol=atol)


def test_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    circuit = cirq.Circuit.from_ops(
        FSWAP(a, b),
        XXYY(a, b),
        YXXY(a, b),
        CCZ(a, b, c),
        CXXYY(a, b, c),
        CYXXY(a, b, c))
    assert circuit.to_text_diagram().strip() == """
a: ───×ᶠ───XXYY───YXXY───@───@──────@──────
      │    │      │      │   │      │
b: ───×ᶠ───XXYY───#2─────@───XXYY───YXXY───
                         │   │      │
c: ──────────────────────Z───XXYY───#2─────
    """.strip()

    circuit = cirq.Circuit.from_ops(
        XXYY(a, b)**0.5,
        YXXY(a, b)**0.5,
        CCZ(a, b, c)**-0.5,
        CXXYY(a, b, c)**-0.5,
        CYXXY(a, b, c)**-0.5)
    assert circuit.to_text_diagram().strip() == """
a: ───XXYY^0.5───YXXY^0.5───@^-0.5───@^-0.5───@^-0.5───
      │          │          │        │        │
b: ───XXYY───────#2─────────@────────XXYY─────YXXY─────
                            │        │        │
c: ─────────────────────────Z────────XXYY─────#2───────
    """.strip()

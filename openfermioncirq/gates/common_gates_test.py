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

import numpy
import pytest
from scipy.linalg import expm, kron

import cirq
from cirq.testing import EqualsTester

from openfermioncirq import FSWAP, XXYY, XXYYGate, YXXY, YXXYGate, ZZ, ZZGate


def test_fswap_interchangeable():
    a, b = cirq.LineQubit(0), cirq.LineQubit(1)
    assert FSWAP(a, b) == FSWAP(b, a)


def test_fswap_inverse():
    assert FSWAP.inverse() == FSWAP


def test_fswap_repr():
    assert repr(FSWAP) == 'FSWAP'


def test_fswap_on_simulator():
    simulator = cirq.google.XmonSimulator()
    a, b = cirq.LineQubit.range(2)
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
    assert XXYYGate(half_turns=0.5).half_turns == 0.5
    assert XXYYGate(half_turns=1.5).half_turns == 1.5
    assert XXYYGate(half_turns=5).half_turns == 1


def test_xxyy_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = XXYYGate(half_turns=1.0, duration=numpy.pi/2)


def test_xxyy_eq():
    eq = EqualsTester()

    eq.add_equality_group(XXYYGate(half_turns=3.5),
                          XXYYGate(half_turns=-0.5),
                          XXYYGate(rads=-0.5 * numpy.pi),
                          XXYYGate(degs=-90),
                          XXYYGate(duration=-0.5 * numpy.pi / 2))

    eq.add_equality_group(XXYYGate(half_turns=1.5),
                          XXYYGate(half_turns=-2.5),
                          XXYYGate(rads=1.5 * numpy.pi),
                          XXYYGate(degs=-450),
                          XXYYGate(duration=-2.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: XXYYGate(half_turns=0))
    eq.make_equality_group(lambda: XXYYGate(half_turns=0.5))


def test_xxyy_interchangeable():
    a, b = cirq.LineQubit(0), cirq.LineQubit(1)
    assert XXYY(a, b) == XXYY(b, a)


def test_xxyy_repr():
    assert repr(XXYYGate(half_turns=1)) == 'XXYY'
    assert repr(XXYYGate(half_turns=0.5)) == 'XXYY**0.5'


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_xxyy_decompose(half_turns):

    gate = XXYY**half_turns
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, gate.matrix(), atol=1e-8)


def test_xxyy_matrix():
    numpy.testing.assert_allclose(XXYYGate(half_turns=2).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(XXYYGate(half_turns=1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, -1j, 0],
                                               [0, -1j, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(XXYYGate(half_turns=0).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(XXYYGate(half_turns=-1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, 1j, 0],
                                               [0, 1j, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    XX = kron(X, X)
    YY = kron(Y, Y)
    numpy.testing.assert_allclose(XXYYGate(half_turns=0.25).matrix(),
                                  expm(-1j * numpy.pi * 0.25 * (XX + YY) / 4))


def test_yxxy_init():
    assert YXXYGate(half_turns=0.5).half_turns == 0.5
    assert YXXYGate(half_turns=1.5).half_turns == 1.5
    assert YXXYGate(half_turns=5).half_turns == 1


def test_yxxy_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = YXXYGate(half_turns=1.0, duration=numpy.pi/2)


def test_yxxy_eq():
    eq = EqualsTester()

    eq.add_equality_group(YXXYGate(half_turns=3.5),
                          YXXYGate(half_turns=-0.5),
                          YXXYGate(rads=-0.5 * numpy.pi),
                          YXXYGate(degs=-90),
                          YXXYGate(duration=-0.5 * numpy.pi / 2))

    eq.add_equality_group(YXXYGate(half_turns=1.5),
                          YXXYGate(half_turns=-2.5),
                          YXXYGate(rads=1.5 * numpy.pi),
                          YXXYGate(degs=-450),
                          YXXYGate(duration=-2.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: YXXYGate(half_turns=0))
    eq.make_equality_group(lambda: YXXYGate(half_turns=0.5))


def test_yxxy_repr():
    assert repr(YXXYGate(half_turns=1)) == 'YXXY'
    assert repr(YXXYGate(half_turns=0.5)) == 'YXXY**0.5'


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_yxxy_decompose(half_turns):

    gate = YXXY**half_turns
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, gate.matrix(), atol=1e-8)


def test_yxxy_matrix():
    numpy.testing.assert_allclose(YXXYGate(half_turns=2).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(YXXYGate(half_turns=1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(YXXYGate(half_turns=0).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(YXXYGate(half_turns=-1).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    YX = kron(Y, X)
    XY = kron(X, Y)
    numpy.testing.assert_allclose(YXXYGate(half_turns=0.25).matrix(),
                                  expm(-1j * numpy.pi * 0.25 * (YX - XY) / 4))


def test_zz_init():
    assert ZZGate(half_turns=0.5).half_turns == 0.5
    assert ZZGate(half_turns=1.5).half_turns == -0.5
    assert ZZGate(half_turns=5).half_turns == 1


def test_zz_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ZZGate(half_turns=1.0, duration=numpy.pi/2)


def test_zz_eq():
    eq = EqualsTester()

    eq.add_equality_group(ZZGate(half_turns=3.5),
                          ZZGate(half_turns=-0.5),
                          ZZGate(rads=-0.5 * numpy.pi),
                          ZZGate(degs=-90),
                          ZZGate(duration=-numpy.pi / 4))

    eq.add_equality_group(ZZGate(half_turns=2.5),
                          ZZGate(half_turns=0.5),
                          ZZGate(rads=0.5 * numpy.pi),
                          ZZGate(degs=90),
                          ZZGate(duration=0.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: ZZGate(half_turns=0))
    eq.make_equality_group(lambda: ZZGate(half_turns=0.1))


def test_zz_repr():
    assert repr(ZZGate(half_turns=1)) == 'ZZ'
    assert repr(ZZGate(half_turns=0.5)) == 'ZZ**0.5'


def test_zz_matrix():
    numpy.testing.assert_allclose(ZZGate(half_turns=0).matrix(),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(ZZGate(half_turns=0.5).matrix(),
                                  numpy.array([[(-1j)**0.5, 0, 0, 0],
                                               [0, 1j**0.5, 0, 0],
                                               [0, 0, 1j**0.5, 0],
                                               [0, 0, 0, (-1j)**0.5]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(ZZGate(half_turns=1).matrix(),
                                  numpy.array([[-1j, 0, 0, 0],
                                               [0, 1j, 0, 0],
                                               [0, 0, 1j, 0],
                                               [0, 0, 0, -1j]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(ZZGate(half_turns=-0.5).matrix(),
                                  numpy.array([[(1j)**0.5, 0, 0, 0],
                                               [0, (-1j)**0.5, 0, 0],
                                               [0, 0, (-1j)**0.5, 0],
                                               [0, 0, 0, (1j)**0.5]]),
                                  atol=1e-8)

    Z = numpy.array([[1, 0], [0, -1]])
    ZZ = kron(Z, Z)
    numpy.testing.assert_allclose(ZZGate(half_turns=0.25).matrix(),
                                  expm(-1j * numpy.pi * 0.25 * ZZ / 2))


@pytest.mark.parametrize(
        'gate, half_turns, initial_state, correct_state, atol', [
            (XXYY, 1.0, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, -1j, -1j, 0]) / numpy.sqrt(2), 1e-7),

            (XXYY, 0.5, numpy.array([1, 1, 0, 0]) / numpy.sqrt(2),
                  numpy.array([1 / numpy.sqrt(2), 0.5, -0.5j, 0]), 1e-7),

            (XXYY, -0.5, numpy.array([1, 1, 0, 0]) / numpy.sqrt(2),
                   numpy.array([1 / numpy.sqrt(2), 0.5, 0.5j, 0]), 1e-7),

            (YXXY, 1.0, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 1, -1, 0]) / numpy.sqrt(2), 1e-7),

            (YXXY, 0.5, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 0, 1, 0]), 1e-7),

            (YXXY, -0.5, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                   numpy.array([0, 1, 0, 0]), 1e-7),

            (ZZ, 1.0, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, -1, -1, 0]) / numpy.sqrt(2), 1e-7),

            (ZZ, 0.5, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 1, 1, 0]) / numpy.sqrt(2), 1e-7),

            (ZZ, -0.5, numpy.array([1, 1, 0, 0]) / numpy.sqrt(2),
                   numpy.array([1, -1j, 0, 0]) / numpy.sqrt(2), 1e-7)
])
def test_two_qubit_rotation_gates_on_simulator(
        gate, half_turns, initial_state, correct_state, atol):
    simulator = cirq.google.XmonSimulator()
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(gate(a, b)**half_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_state, correct_state, atol=atol)


def test_common_gate_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    circuit = cirq.Circuit.from_ops(
        FSWAP(a, b),
        XXYY(a, b),
        YXXY(a, b),
        ZZ(a, b))
    assert circuit.to_text_diagram().strip() == """
a: ───×ᶠ───XXYY───YXXY───Z───
      │    │      │      │
b: ───×ᶠ───XXYY───#2─────Z───
""".strip()

    assert circuit.to_text_diagram(use_unicode_characters=False).strip() == """
a: ---fswap---XXYY---YXXY---Z---
      |       |      |      |
b: ---fswap---XXYY---#2-----Z---
""".strip()

    circuit = cirq.Circuit.from_ops(
        XXYY(a, b)**0.5,
        YXXY(a, b)**0.5,
        ZZ(a, b)**0.5)
    assert circuit.to_text_diagram().strip() == """
a: ───XXYY───────YXXY─────Z───────
      │          │        │
b: ───XXYY^0.5───#2^0.5───Z^0.5───
""".strip()
